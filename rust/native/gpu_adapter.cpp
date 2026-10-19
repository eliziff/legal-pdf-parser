#include <stdint.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

extern "C" int32_t legalpdf_preferred_dml_adapter() {
  ComPtr<IDXGIFactory6> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return -1;
  for (UINT rank = 0;; ++rank) {
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapterByGpuPreference(rank, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                  IID_PPV_ARGS(&adapter)))) break;
    DXGI_ADAPTER_DESC1 description = {};
    if (FAILED(adapter->GetDesc1(&description)) ||
        (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                IID_PPV_ARGS(&device)))) continue;
    D3D12_FEATURE_DATA_ARCHITECTURE1 architecture = {};
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 options = {};
    // Integrated and FP16-emulating devices use CPU by default; explicit GPU selection is retained.
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &architecture,
                                           sizeof(architecture))) || architecture.UMA ||
        FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &options,
                                           sizeof(options))) || !options.Native16BitShaderOpsSupported) continue;
    // DirectML uses EnumAdapters order, which differs from Windows' GPU-preference order.
    for (UINT index = 0;; ++index) {
      ComPtr<IDXGIAdapter1> ordered;
      if (FAILED(factory->EnumAdapters1(index, &ordered))) break;
      DXGI_ADAPTER_DESC1 other = {};
      if (SUCCEEDED(ordered->GetDesc1(&other)) &&
          other.AdapterLuid.HighPart == description.AdapterLuid.HighPart &&
          other.AdapterLuid.LowPart == description.AdapterLuid.LowPart) return static_cast<int32_t>(index);
    }
  }
  return -1;
}
