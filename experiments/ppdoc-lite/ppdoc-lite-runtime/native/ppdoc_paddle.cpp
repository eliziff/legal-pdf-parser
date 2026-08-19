#include "ppdoc_paddle.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "paddle_inference_api.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {
thread_local std::string last_error;

struct Session {
  std::shared_ptr<paddle_infer::Predictor> predictor;
#ifdef _WIN32
  HMODULE mklml = nullptr;
  HMODULE onednn = nullptr;

  ~Session() {
    predictor.reset();
    if (onednn) FreeLibrary(onednn);
    if (mklml) FreeLibrary(mklml);
  }
#endif
};

#ifdef _WIN32
std::wstring sibling(const wchar_t* name) {
  HMODULE module = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&ppdoc_paddle_abi_version),
                          &module)) {
    throw std::runtime_error("could not locate ppdoc_paddle.dll");
  }
  std::vector<wchar_t> path(32768);
  const DWORD length = GetModuleFileNameW(module, path.data(), path.size());
  if (length == 0 || length == path.size()) {
    throw std::runtime_error("could not resolve the Paddle runtime directory");
  }
  std::wstring result(path.data(), length);
  const auto separator = result.find_last_of(L"\\/");
  result.resize(separator == std::wstring::npos ? 0 : separator + 1);
  return result + name;
}

HMODULE load_sibling(const wchar_t* name) {
  auto path = sibling(name);
  auto module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!module) {
    throw std::runtime_error("could not load a required sibling Paddle library");
  }
  return module;
}
#endif

template <typename Function>
int guarded(Function function) noexcept {
  try {
    function();
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "unknown native Paddle error";
  }
  return -1;
}
}  // namespace

int ppdoc_paddle_abi_version(void) { return 1; }

void* ppdoc_paddle_create(const char* model, const char* params, int threads,
                          int enable_onednn) {
  Session* session = nullptr;
  if (guarded([&] {
        if (!model || !params || threads < 0) {
          throw std::invalid_argument("invalid Paddle session arguments");
        }
        session = new Session{};
#ifdef _WIN32
        session->mklml = load_sibling(L"mklml.dll");
        if (enable_onednn) session->onednn = load_sibling(L"mkldnn.dll");
#endif
        paddle_infer::Config config(model, params);
        config.DisableGpu();
        config.SetCpuMathLibraryNumThreads(std::max(1, threads));
        config.SwitchIrOptim(true);
        config.DisableGlogInfo();
        if (enable_onednn) {
#ifdef PPDOC_LEGACY_ONEDNN
          config.EnableMKLDNN();
          config.SetMkldnnCacheCapacity(1);
#else
          config.EnableONEDNN();
          config.SetOnednnCacheCapacity(1);
#endif
        }
        session->predictor = paddle_infer::CreatePredictor(config);
        if (!session->predictor) throw std::runtime_error("could not create Paddle predictor");
      }) != 0) {
    delete session;
    return nullptr;
  }
  return session;
}

int ppdoc_paddle_run(void* handle, const float* image, int target_height,
                     int target_width, const float* im_shape,
                     const float* scale_factor, float* boxes,
                     size_t boxes_capacity, size_t* boxes_length,
                     int32_t* count) {
  return guarded([&] {
    if (!handle || !image || !im_shape || !scale_factor || !boxes ||
        !boxes_length || !count || target_height <= 0 || target_width <= 0) {
      throw std::invalid_argument("invalid Paddle inference arguments");
    }
    auto& predictor = static_cast<Session*>(handle)->predictor;
    for (const auto& name : predictor->GetInputNames()) {
      auto tensor = predictor->GetInputHandle(name);
      if (name == "image") {
        tensor->Reshape({1, 3, target_height, target_width});
        tensor->CopyFromCpu(image);
      } else if (name == "im_shape") {
        tensor->Reshape({1, 2});
        tensor->CopyFromCpu(im_shape);
      } else if (name == "scale_factor") {
        tensor->Reshape({1, 2});
        tensor->CopyFromCpu(scale_factor);
      } else {
        throw std::runtime_error("unexpected Paddle input: " + name);
      }
    }
    if (!predictor->Run()) throw std::runtime_error("Paddle inference failed");
    const auto names = predictor->GetOutputNames();
    if (names.size() < 2) throw std::runtime_error("Paddle model returned fewer than two outputs");

    auto box_tensor = predictor->GetOutputHandle(names[0]);
    size_t length = 1;
    for (auto dimension : box_tensor->shape()) length *= static_cast<size_t>(dimension);
    if (length > boxes_capacity) throw std::runtime_error("Paddle box buffer is too small");
    box_tensor->CopyToCpu(boxes);
    *boxes_length = length;

    auto count_tensor = predictor->GetOutputHandle(names[1]);
    if (count_tensor->type() == paddle_infer::INT32) {
      count_tensor->CopyToCpu(count);
    } else if (count_tensor->type() == paddle_infer::INT64) {
      int64_t value = 0;
      count_tensor->CopyToCpu(&value);
      *count = static_cast<int32_t>(value);
    } else {
      throw std::runtime_error("Paddle count output is not integer");
    }
  });
}

const char* ppdoc_paddle_last_error(void) { return last_error.c_str(); }

void ppdoc_paddle_destroy(void* handle) { delete static_cast<Session*>(handle); }
