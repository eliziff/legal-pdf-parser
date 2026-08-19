param(
  [ValidateSet('heron-int8')][string]$Model = 'heron-int8',
  [Parameter(Mandatory = $true)][string]$OutputDir,
  [string]$SourceFile,
  [switch]$Force
)

$ErrorActionPreference = 'Stop'
$sources = @{
  'heron-int8' = @{
    Url = 'https://github.com/docling-project/docling.rs/releases/download/models-v1/layout_heron_int8.onnx'
    Sha256 = '5c7a4685c838b485069b81847f2c9330f7ffc488aefff7a8ceb7f7968c95e410'
    Manifest = Join-Path $PSScriptRoot 'model_manifests/heron-int8.json'
    Notice = Join-Path $PSScriptRoot 'model_manifests/heron-int8.NOTICE.md'
  }
}
$source = $sources[$Model]
$destination = [System.IO.Path]::GetFullPath($OutputDir)
$modelPath = Join-Path $destination 'model.onnx'
$manifestPath = Join-Path $destination 'manifest.json'
if (Test-Path -LiteralPath $destination) {
  if (-not $Force -and (Get-ChildItem -LiteralPath $destination -Force | Select-Object -First 1)) {
    throw "Output directory must be empty (or pass -Force): $destination"
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
    throw "Layout model checksum mismatch: expected $($source.Sha256), found $actual"
  }
  Move-Item -LiteralPath $downloadPath -Destination $modelPath -Force
  Copy-Item -LiteralPath $source.Manifest -Destination $manifestPath -Force
  Copy-Item -LiteralPath $source.Notice -Destination (Join-Path $destination 'NOTICE.md') -Force
} finally {
  if (Test-Path -LiteralPath $downloadPath) {
    Remove-Item -LiteralPath $downloadPath -Force
  }
}

Write-Output $destination
