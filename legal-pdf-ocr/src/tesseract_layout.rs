use crate::kraken::LineBox;
use image::GrayImage;
use legal_pdf_core::{Error, Result};
use libloading::Library;
use std::ffi::c_void;
use std::path::Path;

type Create = unsafe extern "C" fn() -> *mut c_void;
type Delete = unsafe extern "C" fn(*mut c_void);
type Init = unsafe extern "C" fn(*mut c_void);
type SetMode = unsafe extern "C" fn(*mut c_void, i32);
type SetImage = unsafe extern "C" fn(*mut c_void, *const u8, i32, i32, i32, i32);
type SetResolution = unsafe extern "C" fn(*mut c_void, i32);
type Analyse = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type Begin = unsafe extern "C" fn(*mut c_void);
type BlockType = unsafe extern "C" fn(*const c_void) -> i32;
type BoundingBox =
    unsafe extern "C" fn(*const c_void, i32, *mut i32, *mut i32, *mut i32, *mut i32) -> i32;
type Next = unsafe extern "C" fn(*mut c_void, i32) -> i32;
type ThinLines =
    unsafe extern "C" fn(*mut c_void, *const u8, i32, i32, i32, i32, i32, *mut i32, i32) -> i32;

#[repr(C)]
struct Functions {
    set_image: SetImage,
    set_resolution: SetResolution,
    analyse: Analyse,
    begin: Begin,
    block_type: BlockType,
    bounding_box: BoundingBox,
    next: Next,
    delete_iterator: Delete,
}

unsafe extern "C" {
    fn legalpdf_tesseract_lines(
        functions: *const Functions,
        api: *mut c_void,
        pixels: *const u8,
        width: i32,
        height: i32,
        channels: i32,
        stride: i32,
        resolution: i32,
        boxes: *mut i32,
        capacity: i32,
    ) -> i32;
}

pub(crate) struct TesseractLayout {
    api: *mut c_void,
    delete: Delete,
    functions: Option<Functions>,
    thin_lines: Option<ThinLines>,
    dpi: i32,
    _library: Library,
}

// SAFETY: each API handle is independent and the pool lends it to only one
// scoped worker at a time through an exclusive mutable reference.
unsafe impl Send for TesseractLayout {}

impl TesseractLayout {
    pub(crate) fn new(path: &Path, dpi: u16) -> Result<Self> {
        // SAFETY: The library stays owned by Self for longer than every copied symbol.
        let library = unsafe { load_library(path) }.map_err(|error| {
            Error::Message(format!(
                "could not load Tesseract layout library {}: {error}",
                path.display()
            ))
        })?;
        // SAFETY: These signatures are the stable Tesseract C API declared in capi.h.
        unsafe {
            if let Ok(create) = library.get::<Create>(b"legalpdf_layout_create\0") {
                let api = create();
                if api.is_null() {
                    return Err(Error::Message(
                        "Tesseract could not create a layout session".to_owned(),
                    ));
                }
                return Ok(Self {
                    api,
                    delete: symbol(&library, b"legalpdf_layout_destroy\0", path)?,
                    functions: None,
                    thin_lines: Some(symbol(&library, b"legalpdf_layout_lines\0", path)?),
                    dpi: i32::from(dpi),
                    _library: library,
                });
            }
            let create: Create = symbol(&library, b"TessBaseAPICreate\0", path)?;
            let init: Init = symbol(&library, b"TessBaseAPIInitForAnalysePage\0", path)?;
            let set_mode: SetMode = symbol(&library, b"TessBaseAPISetPageSegMode\0", path)?;
            let api = create();
            if api.is_null() {
                return Err(Error::Message(
                    "Tesseract could not create a layout session".to_owned(),
                ));
            }
            init(api);
            set_mode(api, 3); // PSM_AUTO
            Ok(Self {
                api,
                delete: symbol(&library, b"TessBaseAPIDelete\0", path)?,
                functions: Some(Functions {
                    set_image: symbol(&library, b"TessBaseAPISetImage\0", path)?,
                    set_resolution: symbol(&library, b"TessBaseAPISetSourceResolution\0", path)?,
                    analyse: symbol(&library, b"TessBaseAPIAnalyseLayout\0", path)?,
                    begin: symbol(&library, b"TessPageIteratorBegin\0", path)?,
                    block_type: symbol(&library, b"TessPageIteratorBlockType\0", path)?,
                    bounding_box: symbol(&library, b"TessPageIteratorBoundingBox\0", path)?,
                    next: symbol(&library, b"TessPageIteratorNext\0", path)?,
                    delete_iterator: symbol(&library, b"TessPageIteratorDelete\0", path)?,
                }),
                thin_lines: None,
                dpi: i32::from(dpi),
                _library: library,
            })
        }
    }

    pub(crate) fn lines(&mut self, image: &GrayImage) -> Result<Vec<LineBox>> {
        self.lines_pixels(image.as_raw(), image.width(), image.height(), 1)
    }

    pub(crate) fn lines_rgba(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<LineBox>> {
        self.lines_pixels(pixels, width, height, 4)
    }

    fn lines_pixels(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: i32,
    ) -> Result<Vec<LineBox>> {
        let expected = width as usize * height as usize * channels as usize;
        if pixels.len() != expected {
            return Err(Error::Message(
                "Tesseract page buffer has invalid dimensions".to_owned(),
            ));
        }
        let width = i32::try_from(width)
            .map_err(|_| Error::Message("page image is too wide for Tesseract".to_owned()))?;
        let height = i32::try_from(height)
            .map_err(|_| Error::Message("page image is too tall for Tesseract".to_owned()))?;
        let mut raw_boxes = vec![[0_i32; 4]; 128];
        let stride = width
            .checked_mul(channels)
            .ok_or_else(|| Error::Message("Tesseract page stride overflowed".to_owned()))?;
        let mut count = self.raw_lines(pixels, width, height, channels, stride, &mut raw_boxes);
        if count < 0 {
            raw_boxes.resize((-count) as usize, [0; 4]);
            count = self.raw_lines(pixels, width, height, channels, stride, &mut raw_boxes);
        }
        let boxes = raw_boxes
            .into_iter()
            .take(count.max(0) as usize)
            .filter_map(|[left, top, right, bottom]| {
                (right > left && bottom > top).then_some(LineBox {
                    left: left.max(0) as usize,
                    top: top.max(0) as usize,
                    right: right.clamp(0, width) as usize,
                    bottom: bottom.clamp(0, height) as usize,
                })
            })
            .collect();
        Ok(order_footnotes(boxes, width as usize, height as usize)
            .into_iter()
            .map(|bbox| LineBox {
                left: bbox.left.saturating_sub(10),
                top: bbox.top.saturating_sub(6),
                right: (bbox.right + 11).min(width as usize),
                bottom: (bbox.bottom + 7).min(height as usize),
            })
            .collect())
    }

    fn raw_lines(
        &mut self,
        pixels: &[u8],
        width: i32,
        height: i32,
        channels: i32,
        stride: i32,
        boxes: &mut [[i32; 4]],
    ) -> i32 {
        // SAFETY: both buffers outlive this synchronous call; the API and function
        // pointers came from the live Tesseract library owned by self.
        unsafe {
            if let Some(lines) = self.thin_lines {
                lines(
                    self.api,
                    pixels.as_ptr(),
                    width,
                    height,
                    channels,
                    stride,
                    self.dpi,
                    boxes.as_mut_ptr().cast(),
                    boxes.len() as i32,
                )
            } else {
                legalpdf_tesseract_lines(
                    self.functions.as_ref().expect("raw Tesseract functions"),
                    self.api,
                    pixels.as_ptr(),
                    width,
                    height,
                    channels,
                    stride,
                    self.dpi,
                    boxes.as_mut_ptr().cast(),
                    boxes.len() as i32,
                )
            }
        }
    }
}

#[cfg(windows)]
unsafe fn load_library(path: &Path) -> std::result::Result<Library, libloading::Error> {
    use libloading::os::windows::{Library as WindowsLibrary, LOAD_WITH_ALTERED_SEARCH_PATH};
    // SAFETY: Forwarded to the caller; the altered path also resolves sibling DLL dependencies.
    unsafe { WindowsLibrary::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH) }.map(Into::into)
}

#[cfg(not(windows))]
unsafe fn load_library(path: &Path) -> std::result::Result<Library, libloading::Error> {
    // SAFETY: Forwarded to the caller.
    unsafe { Library::new(path) }
}

impl Drop for TesseractLayout {
    fn drop(&mut self) {
        // SAFETY: api was created by this library and has not been deleted.
        unsafe { (self.delete)(self.api) }
    }
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8], path: &Path) -> Result<T> {
    // SAFETY: The caller supplies the C declaration matching this exported name.
    unsafe { library.get::<T>(name) }
        .map(|value| *value)
        .map_err(|error| {
            Error::Message(format!(
                "Tesseract layout library {} is missing {}: {error}",
                path.display(),
                String::from_utf8_lossy(&name[..name.len() - 1])
            ))
        })
}

fn median(mut values: Vec<usize>) -> usize {
    values.sort_unstable();
    values[values.len() / 2]
}

fn footer_start(mut lines: Vec<LineBox>, page_height: usize) -> Option<usize> {
    lines.sort_by_key(|line| line.top);
    if lines.len() < 8 {
        return None;
    }
    let body_end = (lines.len() * 65).div_ceil(100);
    let body_height = median(lines[..body_end].iter().map(|line| line.height()).collect());
    for index in (lines.len() * 45).div_ceil(100)..lines.len() - 2 {
        let gap = lines[index].top.saturating_sub(lines[index - 1].bottom);
        let tail_height = median(lines[index..].iter().map(|line| line.height()).collect());
        if lines[index].top > page_height * 55 / 100
            && gap >= 12_usize.max(body_height * 3 / 4)
            && tail_height * 100 <= body_height * 92
        {
            return Some(lines[index].top);
        }
    }
    None
}

// Tesseract's order is retained except for its known two-column footnote behavior.
fn order_footnotes(boxes: Vec<LineBox>, page_width: usize, page_height: usize) -> Vec<LineBox> {
    if boxes.len() < 16 {
        return boxes;
    }
    let center = |line: &LineBox| (line.left + line.right) as f64 / 2.0;
    let left = boxes
        .iter()
        .copied()
        .filter(|line| center(line) < page_width as f64 * 0.48)
        .collect::<Vec<_>>();
    let right = boxes
        .iter()
        .copied()
        .filter(|line| center(line) > page_width as f64 * 0.52)
        .collect::<Vec<_>>();
    let (Some(left_start), Some(right_start)) = (
        footer_start(left, page_height),
        footer_start(right, page_height),
    ) else {
        return boxes;
    };
    if left_start.abs_diff(right_start) > page_height * 8 / 100 {
        return boxes;
    }
    let is_footer = |line: &LineBox| {
        let middle = center(line);
        (middle < page_width as f64 * 0.48 && line.top >= left_start)
            || (middle > page_width as f64 * 0.52 && line.top >= right_start)
    };
    let mut output = boxes
        .iter()
        .copied()
        .filter(|line| !is_footer(line))
        .collect::<Vec<_>>();
    let mut footnotes = boxes.into_iter().filter(is_footer).collect::<Vec<_>>();
    footnotes.sort_by_key(|line| (center(line) > page_width as f64 / 2.0, line.top, line.left));
    output.extend(footnotes);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_column_footnotes_follow_both_body_columns() {
        let line = |left, top, height| LineBox {
            left,
            top,
            right: left + 360,
            bottom: top + height,
        };
        let left_body = (0..12)
            .map(|i| line(75, 120 + i * 25, 20))
            .collect::<Vec<_>>();
        let left_notes = (0..4)
            .map(|i| line(92, 500 + i * 20, 16))
            .collect::<Vec<_>>();
        let right_body = (0..12)
            .map(|i| line(565, 120 + i * 25, 20))
            .collect::<Vec<_>>();
        let right_notes = (0..4)
            .map(|i| line(580, 500 + i * 20, 16))
            .collect::<Vec<_>>();
        let input = [
            left_body.clone(),
            left_notes.clone(),
            right_body.clone(),
            right_notes.clone(),
        ]
        .concat();
        let expected = [left_body, right_body, left_notes, right_notes].concat();
        assert_eq!(order_footnotes(input, 1000, 700), expected);
    }
}
