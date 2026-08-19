#include <tesseract/baseapi.h>
#include <tesseract/pageiterator.h>

#define EXPORT extern "C" __declspec(dllexport)

EXPORT tesseract::TessBaseAPI *legalpdf_layout_create() {
  auto *api = new tesseract::TessBaseAPI();
  api->InitForAnalysePage();
  api->SetPageSegMode(tesseract::PSM_AUTO);
  return api;
}

EXPORT void legalpdf_layout_destroy(tesseract::TessBaseAPI *api) { delete api; }

EXPORT int legalpdf_layout_lines(tesseract::TessBaseAPI *api,
                                 const unsigned char *pixels, int width,
                                 int height, int channels, int stride,
                                 int resolution, int *boxes, int capacity) {
  api->SetImage(pixels, width, height, channels, stride);
  api->SetSourceResolution(resolution);
  auto *iterator = api->AnalyseLayout();
  if (!iterator) return 0;

  int count = 0;
  iterator->Begin();
  do {
    int left, top, right, bottom;
    const int block = iterator->BlockType();
    if (block >= tesseract::PT_FLOWING_TEXT &&
        block <= tesseract::PT_CAPTION_TEXT &&
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
