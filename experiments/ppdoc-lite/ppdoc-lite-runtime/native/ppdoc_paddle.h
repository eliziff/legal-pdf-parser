#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define PPDOC_API __declspec(dllexport)
#else
#define PPDOC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

PPDOC_API int ppdoc_paddle_abi_version(void);
PPDOC_API void* ppdoc_paddle_create(const char* model, const char* params,
                                    int threads, int enable_onednn);
PPDOC_API int ppdoc_paddle_run(void* handle, const float* image,
                              int target_height, int target_width,
                              const float* im_shape, const float* scale_factor,
                              float* boxes, size_t boxes_capacity,
                              size_t* boxes_length, int32_t* count);
PPDOC_API const char* ppdoc_paddle_last_error(void);
PPDOC_API void ppdoc_paddle_destroy(void* handle);

#ifdef __cplusplus
}
#endif
