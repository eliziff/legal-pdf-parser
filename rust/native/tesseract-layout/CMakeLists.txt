cmake_minimum_required(VERSION 3.20)
project(legalpdf_tesseract_layout LANGUAGES CXX)

add_library(legalpdf_tesseract_layout SHARED layout_core.cpp)
target_compile_features(legalpdf_tesseract_layout PRIVATE cxx_std_17)
target_compile_options(legalpdf_tesseract_layout PRIVATE /O2 /GL)
target_link_options(legalpdf_tesseract_layout PRIVATE /LTCG /OPT:REF /OPT:ICF)
target_compile_definitions(legalpdf_tesseract_layout PRIVATE TESS_EXPORTS)
target_include_directories(legalpdf_tesseract_layout PRIVATE
  "${TESSERACT_SOURCE}/include"
  "${TESSERACT_SOURCE}/src"
  "${TESSERACT_BUILD}"
  "${TESSERACT_BUILD}/include"
  "${LEPTONICA_SOURCE}/src"
  "${LEPTONICA_BUILD}/src")
target_link_libraries(legalpdf_tesseract_layout PRIVATE
  "${TESSERACT_LIBRARY}"
  "${LEPTONICA_LIBRARY}")
