use crate::ppdoc::{input_kind, ModelInput};
use legal_pdf_core::{Error, Result};
use libloading::Library;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::ptr;

const OV_OK: i32 = 0;
const OV_F32: u32 = 4;
const OV_I32: u32 = 9;
const OV_I64: u32 = 10;

type OvCoreCreate = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type OvCoreFree = unsafe extern "C" fn(*mut c_void);
type OvCoreCompileModelFromFile = unsafe extern "C" fn(
    *const c_void,
    *const c_char,
    *const c_char,
    usize,
    *mut *mut c_void,
    ...
) -> i32;
type OvCompiledModelCreateInferRequest =
    unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> i32;
type OvCompiledModelFree = unsafe extern "C" fn(*mut c_void);
type OvInferRequestSetTensor =
    unsafe extern "C" fn(*mut c_void, *const c_char, *const c_void) -> i32;
type OvInferRequestGetTensor =
    unsafe extern "C" fn(*const c_void, *const c_char, *mut *mut c_void) -> i32;
type OvInferRequestInfer = unsafe extern "C" fn(*mut c_void) -> i32;
type OvInferRequestFree = unsafe extern "C" fn(*mut c_void);
type OvTensorCreateFromHostPtr =
    unsafe extern "C" fn(u32, OvShape, *mut c_void, *mut *mut c_void) -> i32;
type OvTensorGetElementType = unsafe extern "C" fn(*const c_void, *mut u32) -> i32;
type OvTensorGetShape = unsafe extern "C" fn(*const c_void, *mut OvShape) -> i32;
type OvTensorGetSize = unsafe extern "C" fn(*const c_void, *mut usize) -> i32;
type OvTensorData = unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> i32;
type OvTensorFree = unsafe extern "C" fn(*mut c_void);
type OvShapeFree = unsafe extern "C" fn(*mut OvShape) -> i32;
type OvGetLastError = unsafe extern "C" fn() -> *const c_char;
type OvFree = unsafe extern "C" fn(*const c_char);

#[repr(C)]
#[derive(Clone, Copy)]
struct OvShape {
    rank: i64,
    dims: *mut i64,
}

#[derive(Clone, Copy)]
struct OpenVinoFunctions {
    core_create: OvCoreCreate,
    core_free: OvCoreFree,
    compile_model: OvCoreCompileModelFromFile,
    create_request: OvCompiledModelCreateInferRequest,
    compiled_model_free: OvCompiledModelFree,
    set_tensor: OvInferRequestSetTensor,
    get_tensor: OvInferRequestGetTensor,
    infer: OvInferRequestInfer,
    request_free: OvInferRequestFree,
    tensor_create: OvTensorCreateFromHostPtr,
    tensor_element_type: OvTensorGetElementType,
    tensor_shape: OvTensorGetShape,
    tensor_size: OvTensorGetSize,
    tensor_data: OvTensorData,
    tensor_free: OvTensorFree,
    shape_free: OvShapeFree,
    last_error: OvGetLastError,
    free: OvFree,
}

impl OpenVinoFunctions {
    unsafe fn load(library: &Library, runtime: &Path) -> Result<Self> {
        Ok(Self {
            core_create: openvino_symbol(library, b"ov_core_create\0", runtime)?,
            core_free: openvino_symbol(library, b"ov_core_free\0", runtime)?,
            compile_model: openvino_symbol(library, b"ov_core_compile_model_from_file\0", runtime)?,
            create_request: openvino_symbol(
                library,
                b"ov_compiled_model_create_infer_request\0",
                runtime,
            )?,
            compiled_model_free: openvino_symbol(library, b"ov_compiled_model_free\0", runtime)?,
            set_tensor: openvino_symbol(library, b"ov_infer_request_set_tensor\0", runtime)?,
            get_tensor: openvino_symbol(library, b"ov_infer_request_get_tensor\0", runtime)?,
            infer: openvino_symbol(library, b"ov_infer_request_infer\0", runtime)?,
            request_free: openvino_symbol(library, b"ov_infer_request_free\0", runtime)?,
            tensor_create: openvino_symbol(library, b"ov_tensor_create_from_host_ptr\0", runtime)?,
            tensor_element_type: openvino_symbol(
                library,
                b"ov_tensor_get_element_type\0",
                runtime,
            )?,
            tensor_shape: openvino_symbol(library, b"ov_tensor_get_shape\0", runtime)?,
            tensor_size: openvino_symbol(library, b"ov_tensor_get_size\0", runtime)?,
            tensor_data: openvino_symbol(library, b"ov_tensor_data\0", runtime)?,
            tensor_free: openvino_symbol(library, b"ov_tensor_free\0", runtime)?,
            shape_free: openvino_symbol(library, b"ov_shape_free\0", runtime)?,
            last_error: openvino_symbol(library, b"ov_get_last_err_msg\0", runtime)?,
            free: openvino_symbol(library, b"ov_free\0", runtime)?,
        })
    }

    fn check(self, status: i32, operation: &str) -> Result<()> {
        if status == OV_OK {
            return Ok(());
        }
        let detail = unsafe {
            let message = (self.last_error)();
            if message.is_null() {
                None
            } else {
                let detail = CStr::from_ptr(message).to_string_lossy().into_owned();
                (self.free)(message);
                Some(detail)
            }
        };
        Err(Error::Message(match detail {
            Some(detail) if !detail.is_empty() => {
                format!("PPdoc OpenVINO {operation} failed ({status}): {detail}")
            }
            _ => format!("PPdoc OpenVINO {operation} failed ({status})"),
        }))
    }
}

pub(crate) struct OpenVinoRawOutputs {
    pub boxes_shape: Vec<i64>,
    pub boxes: Vec<f32>,
    pub logits_shape: Vec<i64>,
    pub logits: Vec<f32>,
}

pub(crate) struct OpenVinoDecodedOutputs {
    pub boxes_shape: Vec<i64>,
    pub boxes: Vec<f32>,
    pub count: Option<usize>,
}

pub(crate) struct OpenVinoSession {
    functions: OpenVinoFunctions,
    core: *mut c_void,
    compiled_model: *mut c_void,
    request: *mut c_void,
    _library: Library,
}

impl OpenVinoSession {
    pub(crate) fn new(
        runtime: &Path,
        model: &Path,
        device: &str,
        requested_threads: usize,
        cache_dir: Option<&Path>,
    ) -> Result<Self> {
        let library = unsafe { load_library(runtime) }.map_err(|error| {
            Error::Message(format!(
                "could not load OpenVINO C runtime {}: {error}",
                runtime.display()
            ))
        })?;
        let functions = unsafe { OpenVinoFunctions::load(&library, runtime)? };
        let mut core = ptr::null_mut();
        functions.check(
            unsafe { (functions.core_create)(&mut core) },
            "core creation",
        )?;

        let device = if device == "default" { "CPU" } else { device };
        let model_path = path_string(model, "model")?;
        let device = CString::new(device)
            .map_err(|_| Error::Message("OpenVINO device contains a NUL byte".to_owned()))?;
        let mut compiled_model = ptr::null_mut();
        let performance_key = c"PERFORMANCE_HINT";
        let latency = c"LATENCY";
        let precision_key = c"INFERENCE_PRECISION_HINT";
        let f32_precision = c"f32";
        let streams_key = c"NUM_STREAMS";
        let one_stream = c"1";
        let cache_dir = cache_dir
            .map(|path| path_string(path, "cache directory"))
            .transpose()?;
        let compile_status = if device.as_bytes() == b"CPU" {
            let threads = if requested_threads == 0 {
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1)
            } else {
                requested_threads
            };
            let threads = CString::new(threads.to_string()).expect("digits contain no NUL");
            unsafe {
                if let Some(cache_dir) = cache_dir.as_ref() {
                    (functions.compile_model)(
                        core,
                        model_path.as_ptr(),
                        device.as_ptr(),
                        16,
                        &mut compiled_model,
                        performance_key.as_ptr(),
                        latency.as_ptr(),
                        precision_key.as_ptr(),
                        f32_precision.as_ptr(),
                        streams_key.as_ptr(),
                        one_stream.as_ptr(),
                        c"INFERENCE_NUM_THREADS".as_ptr(),
                        threads.as_ptr(),
                        c"SCHEDULING_CORE_TYPE".as_ptr(),
                        c"ANY_CORE".as_ptr(),
                        c"ENABLE_CPU_PINNING".as_ptr(),
                        c"NO".as_ptr(),
                        c"ENABLE_HYPER_THREADING".as_ptr(),
                        c"YES".as_ptr(),
                        c"CACHE_DIR".as_ptr(),
                        cache_dir.as_ptr(),
                    )
                } else {
                    (functions.compile_model)(
                        core,
                        model_path.as_ptr(),
                        device.as_ptr(),
                        14,
                        &mut compiled_model,
                        performance_key.as_ptr(),
                        latency.as_ptr(),
                        precision_key.as_ptr(),
                        f32_precision.as_ptr(),
                        streams_key.as_ptr(),
                        one_stream.as_ptr(),
                        c"INFERENCE_NUM_THREADS".as_ptr(),
                        threads.as_ptr(),
                        c"SCHEDULING_CORE_TYPE".as_ptr(),
                        c"ANY_CORE".as_ptr(),
                        c"ENABLE_CPU_PINNING".as_ptr(),
                        c"NO".as_ptr(),
                        c"ENABLE_HYPER_THREADING".as_ptr(),
                        c"YES".as_ptr(),
                    )
                }
            }
        } else {
            unsafe {
                if let Some(cache_dir) = cache_dir.as_ref() {
                    (functions.compile_model)(
                        core,
                        model_path.as_ptr(),
                        device.as_ptr(),
                        8,
                        &mut compiled_model,
                        performance_key.as_ptr(),
                        latency.as_ptr(),
                        precision_key.as_ptr(),
                        f32_precision.as_ptr(),
                        streams_key.as_ptr(),
                        one_stream.as_ptr(),
                        c"CACHE_DIR".as_ptr(),
                        cache_dir.as_ptr(),
                    )
                } else {
                    (functions.compile_model)(
                        core,
                        model_path.as_ptr(),
                        device.as_ptr(),
                        6,
                        &mut compiled_model,
                        performance_key.as_ptr(),
                        latency.as_ptr(),
                        precision_key.as_ptr(),
                        f32_precision.as_ptr(),
                        streams_key.as_ptr(),
                        one_stream.as_ptr(),
                    )
                }
            }
        };
        if let Err(error) = functions.check(compile_status, "model compilation") {
            unsafe { (functions.core_free)(core) };
            return Err(error);
        }

        let mut request = ptr::null_mut();
        if let Err(error) = functions.check(
            unsafe { (functions.create_request)(compiled_model, &mut request) },
            "infer-request creation",
        ) {
            unsafe {
                (functions.compiled_model_free)(compiled_model);
                (functions.core_free)(core);
            }
            return Err(error);
        }
        Ok(Self {
            functions,
            core,
            compiled_model,
            request,
            _library: library,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_raw(
        &mut self,
        inputs: &[String],
        boxes_name: &str,
        logits_name: &str,
        mut pixels: Vec<f32>,
        target_height: usize,
        target_width: usize,
        mut im_shape: [f32; 2],
        mut scale_factor: [f32; 2],
    ) -> Result<OpenVinoRawOutputs> {
        self.infer(
            inputs,
            &mut pixels,
            target_height,
            target_width,
            &mut im_shape,
            &mut scale_factor,
        )?;
        let (boxes_shape, boxes) = self.output_f32(boxes_name)?;
        let (logits_shape, logits) = self.output_f32(logits_name)?;
        Ok(OpenVinoRawOutputs {
            boxes_shape,
            boxes,
            logits_shape,
            logits,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_decoded(
        &mut self,
        inputs: &[String],
        boxes_name: &str,
        counts_name: Option<&str>,
        mut pixels: Vec<f32>,
        target_height: usize,
        target_width: usize,
        mut im_shape: [f32; 2],
        mut scale_factor: [f32; 2],
    ) -> Result<OpenVinoDecodedOutputs> {
        self.infer(
            inputs,
            &mut pixels,
            target_height,
            target_width,
            &mut im_shape,
            &mut scale_factor,
        )?;
        let (boxes_shape, boxes) = self.output_f32(boxes_name)?;
        let count = counts_name
            .map(|name| self.output_count(name))
            .transpose()?;
        Ok(OpenVinoDecodedOutputs {
            boxes_shape,
            boxes,
            count,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn infer(
        &mut self,
        inputs: &[String],
        pixels: &mut [f32],
        target_height: usize,
        target_width: usize,
        im_shape: &mut [f32; 2],
        scale_factor: &mut [f32; 2],
    ) -> Result<()> {
        let mut tensors = Vec::with_capacity(inputs.len());
        for name in inputs {
            let tensor = match input_kind(inputs, name) {
                ModelInput::Image => self.host_tensor(
                    &[1, 3, target_height as i64, target_width as i64],
                    pixels.as_mut_ptr(),
                )?,
                ModelInput::Shape => self.host_tensor(&[1, 2], im_shape.as_mut_ptr())?,
                ModelInput::Scale => self.host_tensor(&[1, 2], scale_factor.as_mut_ptr())?,
            };
            let name = CString::new(name.as_str())
                .map_err(|_| Error::Message("OpenVINO tensor name contains NUL".to_owned()))?;
            self.functions.check(
                unsafe { (self.functions.set_tensor)(self.request, name.as_ptr(), tensor.ptr) },
                "input binding",
            )?;
            tensors.push(tensor);
        }
        self.functions
            .check(unsafe { (self.functions.infer)(self.request) }, "inference")?;
        Ok(())
    }

    fn host_tensor(&self, dimensions: &[i64], data: *mut f32) -> Result<TensorHandle> {
        let shape = OvShape {
            rank: dimensions.len() as i64,
            dims: dimensions.as_ptr().cast_mut(),
        };
        let mut tensor = ptr::null_mut();
        self.functions.check(
            unsafe { (self.functions.tensor_create)(OV_F32, shape, data.cast(), &mut tensor) },
            "tensor creation",
        )?;
        Ok(TensorHandle {
            ptr: tensor,
            free: self.functions.tensor_free,
        })
    }

    fn output_tensor(&self, name: &str) -> Result<(TensorHandle, Vec<i64>, usize, u32)> {
        let name = CString::new(name)
            .map_err(|_| Error::Message("OpenVINO tensor name contains NUL".to_owned()))?;
        let mut tensor = ptr::null_mut();
        self.functions.check(
            unsafe { (self.functions.get_tensor)(self.request, name.as_ptr(), &mut tensor) },
            "output lookup",
        )?;
        let tensor = TensorHandle {
            ptr: tensor,
            free: self.functions.tensor_free,
        };
        let mut element_type = 0;
        self.functions.check(
            unsafe { (self.functions.tensor_element_type)(tensor.ptr, &mut element_type) },
            "output-type query",
        )?;
        let mut shape = OvShape {
            rank: 0,
            dims: ptr::null_mut(),
        };
        self.functions.check(
            unsafe { (self.functions.tensor_shape)(tensor.ptr, &mut shape) },
            "output-shape query",
        )?;
        let dimensions = if shape.rank <= 0 || shape.dims.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(shape.dims, shape.rank as usize) }.to_vec()
        };
        let shape_status = unsafe { (self.functions.shape_free)(&mut shape) };
        self.functions.check(shape_status, "output-shape release")?;
        let mut length = 0;
        self.functions.check(
            unsafe { (self.functions.tensor_size)(tensor.ptr, &mut length) },
            "output-size query",
        )?;
        let expected = dimensions.iter().try_fold(1_usize, |product, dimension| {
            usize::try_from(*dimension)
                .ok()
                .and_then(|dimension| product.checked_mul(dimension))
        });
        if expected != Some(length) {
            return Err(Error::Message(format!(
                "PPdoc OpenVINO output has inconsistent shape {dimensions:?} and length {length}"
            )));
        }
        Ok((tensor, dimensions, length, element_type))
    }

    fn output_f32(&self, name: &str) -> Result<(Vec<i64>, Vec<f32>)> {
        let (tensor, dimensions, length, element_type) = self.output_tensor(name)?;
        if element_type != OV_F32 {
            return Err(Error::Message(format!(
                "PPdoc OpenVINO output {name:?} is not f32 (type {element_type})"
            )));
        }
        let mut data = ptr::null_mut();
        self.functions.check(
            unsafe { (self.functions.tensor_data)(tensor.ptr, &mut data) },
            "output-data query",
        )?;
        if data.is_null() && length > 0 {
            return Err(Error::Message(
                "PPdoc OpenVINO returned a null output buffer".to_owned(),
            ));
        }
        let values = unsafe { std::slice::from_raw_parts(data.cast::<f32>(), length) }.to_vec();
        Ok((dimensions, values))
    }

    fn output_count(&self, name: &str) -> Result<usize> {
        let (tensor, dimensions, length, element_type) = self.output_tensor(name)?;
        if length != 1 {
            return Err(Error::Message(format!(
                "PPdoc OpenVINO count output {name:?} has unexpected shape {dimensions:?}"
            )));
        }
        let mut data = ptr::null_mut();
        self.functions.check(
            unsafe { (self.functions.tensor_data)(tensor.ptr, &mut data) },
            "output-data query",
        )?;
        if data.is_null() {
            return Err(Error::Message(
                "PPdoc OpenVINO returned a null count buffer".to_owned(),
            ));
        }
        let count = unsafe {
            match element_type {
                OV_I32 => i64::from(*data.cast::<i32>()),
                OV_I64 => *data.cast::<i64>(),
                _ => {
                    return Err(Error::Message(format!(
                    "PPdoc OpenVINO count output {name:?} is not i32 or i64 (type {element_type})"
                )))
                }
            }
        };
        Ok(usize::try_from(count.max(0)).unwrap_or(usize::MAX))
    }
}

impl Drop for OpenVinoSession {
    fn drop(&mut self) {
        unsafe {
            (self.functions.request_free)(self.request);
            (self.functions.compiled_model_free)(self.compiled_model);
            (self.functions.core_free)(self.core);
        }
    }
}

struct TensorHandle {
    ptr: *mut c_void,
    free: OvTensorFree,
}

impl Drop for TensorHandle {
    fn drop(&mut self) {
        unsafe { (self.free)(self.ptr) };
    }
}

fn path_string(path: &Path, label: &str) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| Error::Message(format!("OpenVINO {label} path contains a NUL byte")))
}

unsafe fn openvino_symbol<T: Copy>(library: &Library, name: &[u8], runtime: &Path) -> Result<T> {
    library
        .get::<T>(name)
        .map(|symbol| *symbol)
        .map_err(|error| {
            Error::Message(format!(
                "OpenVINO runtime {} is missing {}: {error}",
                runtime.display(),
                String::from_utf8_lossy(&name[..name.len().saturating_sub(1)])
            ))
        })
}

#[cfg(windows)]
unsafe fn load_library(path: &Path) -> std::result::Result<Library, libloading::Error> {
    use libloading::os::windows::{Library as WindowsLibrary, LOAD_WITH_ALTERED_SEARCH_PATH};
    WindowsLibrary::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH).map(Into::into)
}

#[cfg(not(windows))]
unsafe fn load_library(path: &Path) -> std::result::Result<Library, libloading::Error> {
    Library::new(path)
}
