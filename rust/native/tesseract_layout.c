#include <stdint.h>

typedef void (*SetImage)(void *, const uint8_t *, int, int, int, int);
typedef void (*SetResolution)(void *, int);
typedef void *(*Analyse)(void *);
typedef void (*Begin)(void *);
typedef int (*BlockType)(const void *);
typedef int (*BoundingBox)(const void *, int, int *, int *, int *, int *);
typedef int (*Next)(void *, int);
typedef void (*Delete)(void *);

typedef struct {
  SetImage set_image;
  SetResolution set_resolution;
  Analyse analyse;
  Begin begin;
  BlockType block_type;
  BoundingBox bounding_box;
  Next next;
  Delete delete_iterator;
} LayoutFunctions;

// Native form of the browser layout-core loop: Tesseract owns segmentation;
// this keeps its fine-grained iterator ABI on one side of the FFI boundary.
int legalpdf_tesseract_lines(const LayoutFunctions *functions, void *api,
                             const uint8_t *pixels, int width, int height,
                             int channels, int stride, int resolution,
                             int32_t *boxes, int capacity) {
  functions->set_image(api, pixels, width, height, channels, stride);
  functions->set_resolution(api, resolution);
  void *iterator = functions->analyse(api);
  if (!iterator) return 0;

  int count = 0;
  functions->begin(iterator);
  do {
    int left, top, right, bottom;
    const int block_type = functions->block_type(iterator);
    if (block_type >= 1 && block_type <= 8 &&
        functions->bounding_box(iterator, 2, &left, &top, &right, &bottom)) {
      if (count < capacity) {
        int32_t *box = boxes + count * 4;
        box[0] = left;
        box[1] = top;
        box[2] = right;
        box[3] = bottom;
      }
      ++count;
    }
  } while (functions->next(iterator, 2));
  functions->delete_iterator(iterator);
  return count <= capacity ? count : -count;
}
