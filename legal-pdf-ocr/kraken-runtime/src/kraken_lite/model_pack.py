from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .hardware import (
    CPU_PROVIDER,
    ExecutionInfo,
    ProviderConfig,
    provider_name,
    resolve_provider_plan,
)


class Session(Protocol):
    execution_info: ExecutionInfo | None

    def run(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Runs a model and returns named outputs."""


@dataclass(slots=True)
class ModelPack:
    root: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ModelPack":
        root = Path(path)
        if root.is_file():
            raise ValueError(
                "Packed .kocr archives are not implemented in this prototype; "
                "pass the unpacked model directory."
            )
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing model manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cls._validate_manifest(manifest)
        pack = cls(root=root, manifest=manifest)
        pack.verify_files()
        return pack

    @staticmethod
    def _validate_manifest(manifest: dict[str, Any]) -> None:
        if manifest.get("format") != "kraken-lite-model/1":
            raise ValueError("Unsupported model-pack format")
        if manifest.get("kind") not in {"blla-segmentation", "recognition"}:
            raise ValueError("Manifest kind must be blla-segmentation or recognition")
        if not isinstance(manifest.get("model"), dict):
            raise ValueError("Manifest has no model object")
        if not manifest["model"].get("file"):
            raise ValueError("Manifest model.file is required")

    def _member_path(self, value: str, *, label: str) -> Path:
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError(f"{label} path must be relative to the model pack")
        root = self.root.resolve()
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} path escapes the model pack: {value!r}") from exc
        return resolved

    @property
    def model_path(self) -> Path:
        return self._member_path(str(self.manifest["model"]["file"]), label="model")

    @property
    def kind(self) -> str:
        return str(self.manifest["kind"])

    @property
    def id(self) -> str:
        return str(self.manifest.get("id", self.root.name))

    @staticmethod
    def _verify_file(path: Path, expected: str | None, *, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        if expected:
            actual = sha256_file(path)
            if actual.lower() != str(expected).lower():
                raise ValueError(
                    f"{label.capitalize()} hash mismatch for {path}: "
                    f"expected {expected}, got {actual}"
                )

    def verify_files(self) -> None:
        self._verify_file(
            self.model_path, self.manifest["model"].get("sha256"), label="model graph"
        )
        codec = self.manifest.get("codec")
        if isinstance(codec, dict) and codec.get("file"):
            self._verify_file(
                self._member_path(str(codec["file"]), label="codec"),
                codec.get("sha256"),
                label="codec",
            )

    def open_session(
        self,
        *,
        device: str = "auto",
        providers: Sequence[str] | None = None,
        intra_threads: int = 1,
        inter_threads: int = 1,
        allow_cpu_fallback: bool = True,
        tensorrt_fp16: bool = False,
        tensorrt_cache: str | None = None,
        cuda_tf32: bool = True,
    ) -> "OnnxSession":
        return OnnxSession(
            self.model_path,
            device=device,
            providers=providers,
            intra_threads=intra_threads,
            inter_threads=inter_threads,
            allow_cpu_fallback=allow_cpu_fallback,
            tensorrt_fp16=tensorrt_fp16,
            tensorrt_cache=tensorrt_cache,
            cuda_tf32=cuda_tf32,
        )


class OnnxSession:
    """Small ONNX Runtime adapter used by segmentation and recognition.

    The graph and numerical contract are identical on every provider. Only the
    tensor forward pass is delegated to CUDA/ROCm/DirectML/CoreML/OpenVINO;
    BLLA geometry and decoding remain deterministic host-side algorithms.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        providers: Sequence[str] | None = None,
        intra_threads: int = 1,
        inter_threads: int = 1,
        allow_cpu_fallback: bool = True,
        tensorrt_fp16: bool = False,
        tensorrt_cache: str | None = None,
        cuda_tf32: bool = True,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                "ONNX Runtime is required for model execution. Install the "
                "'runtime-cpu' extra, the 'runtime-cuda' extra, or use a "
                "packaged native build."
            ) from exc

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(intra_threads))
        opts.inter_op_num_threads = max(1, int(inter_threads))
        opts.execution_mode = (
            ort.ExecutionMode.ORT_PARALLEL
            if opts.inter_op_num_threads > 1
            else ort.ExecutionMode.ORT_SEQUENTIAL
        )
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = tuple(ort.get_available_providers())
        plan = resolve_provider_plan(
            available,
            device=device,
            providers=providers,
            allow_cpu_fallback=allow_cpu_fallback,
            tensorrt_fp16=tensorrt_fp16,
            tensorrt_cache=tensorrt_cache,
            cuda_tf32=cuda_tf32,
        )
        configured: list[ProviderConfig] = list(plan.configured_providers)
        # DirectML requires sequential execution and does not support ORT's
        # memory-pattern optimization. Set those constraints only when DML is
        # actually selected; CUDA and CPU retain the faster defaults above.
        if plan.primary_provider == "DmlExecutionProvider":
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.enable_mem_pattern = False
        fallback_reason: str | None = None
        try:
            session = ort.InferenceSession(
                str(model_path), sess_options=opts, providers=configured
            )
        except Exception as exc:
            # ``auto`` is intentionally robust on machines with a partially
            # installed GPU stack (for example, the CUDA EP is present but a
            # required CUDA DLL is not). Explicit device requests fail loudly.
            if (
                plan.requested_device == "auto"
                and plan.primary_provider != CPU_PROVIDER
                and allow_cpu_fallback
                and CPU_PROVIDER in available
            ):
                fallback_reason = f"{type(exc).__name__}: {exc}"
                configured = [CPU_PROVIDER]
                session = ort.InferenceSession(
                    str(model_path), sess_options=opts, providers=configured
                )
            else:
                raise RuntimeError(
                    f"Could not initialize ONNX Runtime for {plan.requested_device!r} "
                    f"with providers {plan.configured_names}: {exc}"
                ) from exc

        self._session = session
        if not allow_cpu_fallback and hasattr(self._session, "disable_fallback"):
            self._session.disable_fallback()
        self.input_names = [item.name for item in self._session.get_inputs()]
        self.output_names = [item.name for item in self._session.get_outputs()]
        active = tuple(self._session.get_providers())
        primary = active[0] if active else CPU_PROVIDER
        if plan.primary_provider != CPU_PROVIDER and primary != plan.primary_provider:
            message = (
                f"ONNX Runtime configured {plan.primary_provider} but activated "
                f"{primary}; active providers: {active}"
            )
            if not allow_cpu_fallback:
                raise RuntimeError(message)
            fallback_reason = fallback_reason or message
        self.execution_info = ExecutionInfo(
            requested_device=plan.requested_device,
            available_providers=available,
            configured_providers=tuple(provider_name(value) for value in configured),
            active_providers=active,
            primary_provider=primary,
            accelerated=primary != CPU_PROVIDER,
            fallback_reason=fallback_reason,
        )

    def run(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        missing = [name for name in self.input_names if name not in inputs]
        if missing:
            raise ValueError(f"Missing model inputs: {missing}")
        outputs = self._session.run(
            self.output_names, {name: inputs[name] for name in self.input_names}
        )
        return dict(zip(self.output_names, outputs, strict=True))


def available_execution_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ONNX Runtime is not installed") from exc
    return list(ort.get_available_providers())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
