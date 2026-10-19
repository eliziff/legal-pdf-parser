param(
  [ValidateSet('heron-int8', 'doc-orientation')][string]$Model = 'heron-int8',
  [Parameter(Mandatory = $true)][string]$OutputDir,
  [string]$SourceFile,
  [switch]$Force
)

$ErrorActionPreference = 'Stop'
$sources = @{
  'heron-int8' = @{
    Url = 'https://github.com/docling-project/docling.rs/releases/download/models-v1/layout_heron_int8.onnx'
    Sha256 = '5c7a4685c838b485069b81847f2c9330f7ffc488aefff7a8ceb7f7968c95e410'
    File = 'model.onnx'
    Manifest = Join-Path $PSScriptRoot 'model_manifests/heron-int8.json'
    Notice = Join-Path $PSScriptRoot 'model_manifests/heron-int8.NOTICE.md'
    NoticeFile = 'NOTICE.md'
  }
  # Shared page-orientation classifier for native OCR model packs.
  'doc-orientation' = @{
    Url = 'https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx/resolve/7330ab7039123e46af2dc03154b9969aa412c61d/inference.onnx'
    Sha256 = 'af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92'
    File = 'orientation.onnx'
    Notice = Join-Path $PSScriptRoot 'model_manifests/doc-orientation.NOTICE.md'
    NoticeFile = 'orientation.NOTICE.md'
  }
}
$source = $sources[$Model]
$destination = [System.IO.Path]::GetFullPath($OutputDir)
$modelPath = Join-Path $destination $source.File
$manifestPath = Join-Path $destination 'manifest.json'
if (Test-Path -LiteralPath $destination) {
  if (-not $Force -and (Test-Path -LiteralPath $modelPath)) {
    throw "The model is installed already (pass -Force to replace it): $modelPath"
  }
} else {
  New-Item -ItemType Directory -Path $destination | Out-Null
}

$downloadPath = "$modelPath.download"
try {
  if ($SourceFile) {
    Copy-Item -LiteralPath (Resolve-Path -LiteralPath $SourceFile).Path -Destination $downloadPath
  } else {
    Invoke-WebRequest -Uri $source.Url -OutFile $downloadPath -UseBasicParsing
  }
  $actual = (Get-FileHash -LiteralPath $downloadPath -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actual -ne $source.Sha256) {
    throw "Model checksum mismatch: expected $($source.Sha256), found $actual"
  }
  Move-Item -LiteralPath $downloadPath -Destination $modelPath -Force
  if ($source.Manifest) { Copy-Item -LiteralPath $source.Manifest -Destination $manifestPath -Force }
  Copy-Item -LiteralPath $source.Notice -Destination (Join-Path $destination $source.NoticeFile) -Force
} finally {
  if (Test-Path -LiteralPath $downloadPath) {
    Remove-Item -LiteralPath $downloadPath -Force
  }
}

Write-Output $destination
