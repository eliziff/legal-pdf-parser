use crate::error::{Error, Result};
#[cfg(any(feature = "kraken", feature = "ppdoc"))]
use ort::execution_providers::{
    CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProviderDispatch,
    OneDNNExecutionProvider, OpenVINOExecutionProvider, TensorRTExecutionProvider,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrtBackend {
    #[default]
    #[serde(rename = "cpu")]
    Cpu,
    #[serde(rename = "cuda")]
    Cuda,
    #[serde(rename = "tensorrt")]
    TensorRt,
    #[serde(rename = "directml")]
    DirectMl,
    #[serde(rename = "openvino")]
    OpenVino,
    #[serde(rename = "onednn")]
    OneDnn,
}

impl OrtBackend {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "tensorrt" => Some(Self::TensorRt),
            "directml" => Some(Self::DirectMl),
            "openvino" => Some(Self::OpenVino),
            "onednn" => Some(Self::OneDnn),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::TensorRt => "tensorrt",
            Self::DirectMl => "directml",
            Self::OpenVino => "openvino",
            Self::OneDnn => "onednn",
        }
    }

    pub(crate) fn normalized_device(self, value: Option<&str>) -> Result<String> {
        match self {
            Self::Cpu | Self::OneDnn => {
                if value.is_some() {
                    return Err(Error::Message(format!(
                        "{} backend does not accept a device",
                        self.name()
                    )));
                }
                Ok("default".to_owned())
            }
            Self::Cuda | Self::TensorRt | Self::DirectMl => {
                let value = value.unwrap_or("0");
                let device = value.parse::<i32>().map_err(|_| {
                    Error::Message(format!(
                        "{} device must be a nonnegative integer",
                        self.name()
                    ))
                })?;
                if device < 0 {
                    return Err(Error::Message(format!(
                        "{} device must be a nonnegative integer",
                        self.name()
                    )));
                }
                Ok(device.to_string())
            }
            Self::OpenVino => {
                let device = value.unwrap_or("default").trim();
                if device.is_empty() {
                    return Err(Error::Message("OpenVINO device cannot be empty".to_owned()));
                }
                Ok(device.to_owned())
            }
        }
    }

    #[cfg(any(feature = "kraken", feature = "ppdoc"))]
    pub(crate) fn providers(self, device: &str) -> Vec<ExecutionProviderDispatch> {
        let device_id = || device.parse::<i32>().expect("normalized numeric device");
        match self {
            Self::Cpu => Vec::new(),
            Self::Cuda => vec![CUDAExecutionProvider::default()
                .with_device_id(device_id())
                .build()
                .error_on_failure()],
            Self::TensorRt => vec![
                TensorRTExecutionProvider::default()
                    .with_device_id(device_id())
                    .build()
                    .error_on_failure(),
                CUDAExecutionProvider::default()
                    .with_device_id(device_id())
                    .build()
                    .error_on_failure(),
            ],
            Self::DirectMl => vec![DirectMLExecutionProvider::default()
                .with_device_id(device_id())
                .build()
                .error_on_failure()],
            Self::OpenVino => {
                let provider = OpenVINOExecutionProvider::default().with_dynamic_shapes(true);
                let provider = if device == "default" {
                    provider
                } else {
                    provider.with_device_type(device)
                };
                vec![provider.build().error_on_failure()]
            }
            Self::OneDnn => vec![OneDNNExecutionProvider::default()
                .build()
                .error_on_failure()],
        }
    }
}
