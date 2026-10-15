from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeAlias

ProviderOptions: TypeAlias = dict[str, Any]
ProviderConfig: TypeAlias = str | tuple[str, ProviderOptions]

CPU_PROVIDER = "CPUExecutionProvider"

# Conservative default order. TensorRT is not selected automatically because
# engine building and dynamic-shape support vary materially by installation.
# It remains available through ``--device tensorrt``.
_AUTO_PRIORITY = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    CPU_PROVIDER,
)

_DEVICE_PROVIDERS = {
    "cpu": CPU_PROVIDER,
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


def provider_name(provider: ProviderConfig) -> str:
    return provider[0] if isinstance(provider, tuple) else provider


@dataclass(frozen=True, slots=True)
class ProviderPlan:
    requested_device: str
    available_providers: tuple[str, ...]
    configured_providers: tuple[ProviderConfig, ...]
    allow_cpu_fallback: bool

    @property
    def configured_names(self) -> tuple[str, ...]:
        return tuple(provider_name(provider) for provider in self.configured_providers)

    @property
    def primary_provider(self) -> str:
        return self.configured_names[0]

    @property
    def accelerated(self) -> bool:
        return self.primary_provider != CPU_PROVIDER

    def to_dict(self) -> dict[str, Any]:
        return {
            "requestedDevice": self.requested_device,
            "availableProviders": list(self.available_providers),
            "configuredProviders": list(self.configured_names),
            "providerOptions": [
                {
                    "name": provider_name(provider),
                    "options": dict(provider[1]) if isinstance(provider, tuple) else {},
                }
                for provider in self.configured_providers
            ],
            "primaryProvider": self.primary_provider,
            "accelerated": self.accelerated,
            "allowCpuFallback": self.allow_cpu_fallback,
        }


@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    requested_device: str
    available_providers: tuple[str, ...]
    configured_providers: tuple[str, ...]
    active_providers: tuple[str, ...]
    primary_provider: str
    accelerated: bool
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requestedDevice": self.requested_device,
            "availableProviders": list(self.available_providers),
            "configuredProviders": list(self.configured_providers),
            "activeProviders": list(self.active_providers),
            "primaryProvider": self.primary_provider,
            "accelerated": self.accelerated,
            "fallbackReason": self.fallback_reason,
        }


def _parse_device(value: str) -> tuple[str, int]:
    normalized = value.strip().lower()
    if not normalized:
        normalized = "auto"
    if ":" not in normalized:
        return normalized, 0
    kind, raw_id = normalized.split(":", 1)
    if kind not in {"cuda", "rocm", "directml", "dml", "tensorrt"}:
        raise ValueError(f"Device {value!r} does not accept a numeric device id")
    try:
        device_id = int(raw_id)
    except ValueError as exc:
        raise ValueError(f"Invalid device id in {value!r}") from exc
    if device_id < 0:
        raise ValueError("Device id must be non-negative")
    return kind, device_id


def _provider_with_options(
    name: str,
    device_id: int,
    *,
    tensorrt_fp16: bool = False,
    tensorrt_cache: str | None = None,
    cuda_tf32: bool = True,
) -> ProviderConfig:
    if name == "CUDAExecutionProvider":
        return (
            name,
            {
                "device_id": device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
                "use_tf32": "1" if cuda_tf32 else "0",
            },
        )
    if name == "TensorrtExecutionProvider":
        options: ProviderOptions = {
            "device_id": device_id,
            "trt_engine_cache_enable": "1",
            "trt_timing_cache_enable": "1",
        }
        if tensorrt_fp16:
            options["trt_fp16_enable"] = "1"
        if tensorrt_cache:
            options["trt_engine_cache_path"] = tensorrt_cache
        return (name, options)
    if name in {"ROCMExecutionProvider", "DmlExecutionProvider"}:
        return (name, {"device_id": device_id})
    return name


def resolve_provider_plan(
    available_providers: Sequence[str],
    *,
    device: str = "auto",
    providers: Sequence[str] | None = None,
    allow_cpu_fallback: bool = True,
    tensorrt_fp16: bool = False,
    tensorrt_cache: str | None = None,
    cuda_tf32: bool = True,
) -> ProviderPlan:
    """Resolves a stable ONNX Runtime execution-provider policy.

    ``providers`` is the expert escape hatch and takes precedence over
    ``device``. ``device=auto`` chooses the strongest conservative provider
    available and always retains CPU as a fallback. Explicit devices fail when
    unavailable rather than silently pretending to use acceleration.
    """

    available = tuple(str(provider) for provider in available_providers)
    if CPU_PROVIDER not in available:
        # Some test doubles and custom ORT builds omit CPU. Do not invent it.
        allow_cpu_fallback = False

    if providers is not None:
        requested = tuple(str(provider) for provider in providers)
        if not requested:
            raise ValueError("providers cannot be an empty sequence")
        missing = [provider for provider in requested if provider not in available]
        if missing:
            raise RuntimeError(
                f"Requested ONNX Runtime providers are unavailable: {missing}; "
                f"available providers: {list(available)}"
            )
        configured: list[ProviderConfig] = list(requested)
        if allow_cpu_fallback and CPU_PROVIDER not in requested:
            configured.append(CPU_PROVIDER)
        return ProviderPlan(
            requested_device="providers:" + ",".join(requested),
            available_providers=available,
            configured_providers=tuple(configured),
            allow_cpu_fallback=allow_cpu_fallback,
        )

    kind, device_id = _parse_device(device)
    if kind == "auto":
        selected = next((provider for provider in _AUTO_PRIORITY if provider in available), None)
        if selected is None:
            raise RuntimeError(f"No supported ONNX Runtime provider is available: {list(available)}")
        configured = [
            _provider_with_options(
                selected,
                device_id,
                tensorrt_fp16=tensorrt_fp16,
                tensorrt_cache=tensorrt_cache,
                cuda_tf32=cuda_tf32,
            )
        ]
        if selected != CPU_PROVIDER and allow_cpu_fallback and CPU_PROVIDER in available:
            configured.append(CPU_PROVIDER)
        return ProviderPlan(
            requested_device="auto",
            available_providers=available,
            configured_providers=tuple(configured),
            allow_cpu_fallback=allow_cpu_fallback,
        )

    if kind not in _DEVICE_PROVIDERS:
        supported = ", ".join(["auto", *_DEVICE_PROVIDERS])
        raise ValueError(f"Unknown device {device!r}; expected one of {supported}")
    selected = _DEVICE_PROVIDERS[kind]
    if selected not in available:
        raise RuntimeError(
            f"Requested device {device!r} requires {selected}, but available providers are "
            f"{list(available)}"
        )
    configured = [
        _provider_with_options(
            selected,
            device_id,
            tensorrt_fp16=tensorrt_fp16,
            tensorrt_cache=tensorrt_cache,
            cuda_tf32=cuda_tf32,
        )
    ]
    # TensorRT commonly delegates unsupported nodes to CUDA. Keep that work on
    # the same GPU before allowing a final CPU fallback.
    if selected == "TensorrtExecutionProvider" and "CUDAExecutionProvider" in available:
        configured.append(
            _provider_with_options(
                "CUDAExecutionProvider",
                device_id,
                cuda_tf32=cuda_tf32,
            )
        )
    if selected != CPU_PROVIDER and allow_cpu_fallback and CPU_PROVIDER in available:
        configured.append(CPU_PROVIDER)
    return ProviderPlan(
        requested_device=device,
        available_providers=available,
        configured_providers=tuple(configured),
        allow_cpu_fallback=allow_cpu_fallback,
    )
