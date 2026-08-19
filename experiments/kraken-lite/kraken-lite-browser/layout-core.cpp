#include <tesseract/baseapi.h>
#include <tesseract/pageiterator.h>
#include <vector>

extern "C" {

tesseract::TessBaseAPI *kl_create() {
  auto *api = new tesseract::TessBaseAPI();
  api->InitForAnalysePage();
  api->SetPageSegMode(tesseract::PSM_AUTO);
  return api;
}

void kl_destroy(tesseract::TessBaseAPI *api) { delete api; }

void kl_set_psm(tesseract::TessBaseAPI *api, int mode) {
  api->SetPageSegMode(static_cast<tesseract::PageSegMode>(mode));
}

int kl_lines_impl(tesseract::TessBaseAPI *api, const unsigned char *pixels,
                  int width, int height, int channels, int *boxes,
                  int capacity, int resolution) {
  api->SetImage(pixels, width, height, channels, width * channels, 1, 0);
  api->SetSourceResolution(resolution);
  auto *iterator = api->AnalyseLayout();
  if (!iterator) return 0;
  int count = 0;
  iterator->Begin();
  do {
    const int block_type = iterator->BlockType();
    int left, top, right, bottom;
    if (block_type >= tesseract::PT_FLOWING_TEXT &&
        block_type <= tesseract::PT_CAPTION_TEXT &&
        iterator->BoundingBox(tesseract::RIL_TEXTLINE, &left, &top, &right,
                              &bottom)) {
      if (count < capacity) {
        int *box = boxes + count * 4;
        box[0] = left;
        box[1] = top;
        box[2] = right;
        box[3] = bottom;
      }
      ++count;
    }
  } while (iterator->Next(tesseract::RIL_TEXTLINE));
  delete iterator;
  return count <= capacity ? count : -count;
}

int kl_lines(tesseract::TessBaseAPI *api, const unsigned char *rgba, int width,
             int height, int *boxes, int capacity) {
  return kl_lines_impl(api, rgba, width, height, 4, boxes, capacity, 200);
}

int kl_lines_dpi(tesseract::TessBaseAPI *api, const unsigned char *rgba,
                 int width, int height, int *boxes, int capacity,
                 int resolution) {
  return kl_lines_impl(api, rgba, width, height, 4, boxes, capacity,
                       resolution);
}

int kl_lines_binary(tesseract::TessBaseAPI *api, const unsigned char *rgba,
                    int width, int height, int *boxes, int capacity,
                    int resolution, int threshold) {
  const int stride = (width + 7) / 8;
  std::vector<unsigned char> binary(stride * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const unsigned char *pixel = rgba + (y * width + x) * 4;
      if ((pixel[0] * 77 + pixel[1] * 150 + pixel[2] * 29) / 256 <
          threshold) {
        binary[y * stride + x / 8] |= 0x80 >> (x & 7);
      }
    }
  }
  api->SetImage(binary.data(), width, height, 0, stride, 1, 0);
  api->SetSourceResolution(resolution);
  auto *iterator = api->AnalyseLayout();
  if (!iterator) return 0;
  int count = 0;
  iterator->Begin();
  do {
    const int block_type = iterator->BlockType();
    int left, top, right, bottom;
    if (block_type >= tesseract::PT_FLOWING_TEXT &&
        block_type <= tesseract::PT_CAPTION_TEXT &&
        iterator->BoundingBox(tesseract::RIL_TEXTLINE, &left, &top, &right,
                              &bottom)) {
      if (count < capacity) {
        int *box = boxes + count * 4;
        box[0] = left; box[1] = top; box[2] = right; box[3] = bottom;
      }
      ++count;
    }
  } while (iterator->Next(tesseract::RIL_TEXTLINE));
  delete iterator;
  return count <= capacity ? count : -count;
}

int kl_lines_gray(tesseract::TessBaseAPI *api, const unsigned char *gray,
                  int width, int height, int *boxes, int capacity) {
  return kl_lines_impl(api, gray, width, height, 1, boxes, capacity, 200);
}

}
