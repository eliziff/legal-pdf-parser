#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "paddle_inference_api.h"

static std::vector<float> read_f32(const std::string& path, std::size_t count) {
  std::vector<float> values(count);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(values.data()), count * sizeof(float))) {
    throw std::runtime_error("could not read tensor: " + path);
  }
  return values;
}

int main(int argc, char** argv) {
  if (argc < 8 || argc > 10) {
    std::cerr << "usage: paddle_probe MODEL PARAMS IMAGE_F32 WIDTH HEIGHT THREADS RUNS [plain|onednn|ort] [profile]\n";
    return 2;
  }
  const int width = std::stoi(argv[4]);
  const int height = std::stoi(argv[5]);
  const int threads = std::stoi(argv[6]);
  const int runs = std::stoi(argv[7]);
  const std::string backend = argc >= 9 ? argv[8] : "onednn";
  auto image = read_f32(argv[3], 3 * 800 * 800);
  std::vector<float> im_shape{800.0f, 800.0f};
  std::vector<float> scale_factor{800.0f / height, 800.0f / width};

  paddle_infer::Config config(argv[1], argv[2]);
  config.DisableGpu();
  if (backend == "onednn") {
    config.EnableONEDNN();
    config.SetOnednnCacheCapacity(1);
  } else if (backend == "ort") {
    config.EnableONNXRuntime();
    config.EnableORTOptimization();
  } else if (backend != "plain") {
    throw std::runtime_error("backend must be plain, onednn, or ort");
  }
  config.SetCpuMathLibraryNumThreads(threads);
  config.SwitchIrOptim(true);
  // Paddle 3.2 PIR models already use the new executor. EnableMemoryOptim()
  // requests a legacy analysis pass and fails before inference for this graph.
  config.DisableGlogInfo();
  if (argc == 10 && std::string(argv[9]) == "profile") config.EnableProfile();

  const auto load_started = std::chrono::steady_clock::now();
  auto predictor = paddle_infer::CreatePredictor(config);
  const auto loaded = std::chrono::steady_clock::now();
  for (const auto& name : predictor->GetInputNames()) {
    auto tensor = predictor->GetInputHandle(name);
    if (name == "image") {
      tensor->Reshape({1, 3, 800, 800});
      tensor->CopyFromCpu(image.data());
    } else if (name == "im_shape") {
      tensor->Reshape({1, 2});
      tensor->CopyFromCpu(im_shape.data());
    } else if (name == "scale_factor") {
      tensor->Reshape({1, 2});
      tensor->CopyFromCpu(scale_factor.data());
    } else {
      throw std::runtime_error("unexpected input: " + name);
    }
  }

  std::vector<double> elapsed_ms;
  for (int index = 0; index < runs; ++index) {
    const auto started = std::chrono::steady_clock::now();
    if (!predictor->Run()) throw std::runtime_error("Paddle inference failed");
    const auto stopped = std::chrono::steady_clock::now();
    elapsed_ms.push_back(std::chrono::duration<double, std::milli>(stopped - started).count());
  }

  std::cout << "{\"backend\":\"" << backend << "\",\"threads\":" << threads
            << ",\"load_ms\":"
            << std::chrono::duration<double, std::milli>(loaded - load_started).count()
            << ",\"run_ms\":[";
  for (std::size_t index = 0; index < elapsed_ms.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << elapsed_ms[index];
  }
  std::cout << "],\"outputs\":[";
  const auto output_names = predictor->GetOutputNames();
  for (std::size_t index = 0; index < output_names.size(); ++index) {
    if (index) std::cout << ',';
    auto tensor = predictor->GetOutputHandle(output_names[index]);
    const auto shape = tensor->shape();
    std::size_t count = 1;
    for (auto dimension : shape) count *= static_cast<std::size_t>(dimension);
    double first = 0.0;
    if (tensor->type() == paddle_infer::FLOAT32) {
      std::vector<float> values(count);
      tensor->CopyToCpu(values.data());
      if (!values.empty()) first = values[0];
    } else if (tensor->type() == paddle_infer::INT32) {
      std::vector<std::int32_t> values(count);
      tensor->CopyToCpu(values.data());
      if (!values.empty()) first = values[0];
    } else {
      throw std::runtime_error("unsupported output data type");
    }
    std::cout << "{\"name\":\"" << output_names[index] << "\",\"shape\":[";
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
      if (dimension) std::cout << ',';
      std::cout << shape[dimension];
    }
    std::cout << "],\"first\":" << first << '}';
  }
  std::cout << "]}\n";
  return 0;
}
