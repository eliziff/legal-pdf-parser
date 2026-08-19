param(
  [Parameter(Mandatory = $true)][string]$LegalPdfBinary,
  [Parameter(Mandatory = $true)][string]$ModelPack,
  [Parameter(Mandatory = $true)][string]$OpenVinoLibDir,
  [Parameter(Mandatory = $true)][string]$OutputDir,
  [switch]$Gpu
)

$ErrorActionPreference = 'Stop'
$binary = (Resolve-Path -LiteralPath $LegalPdfBinary).Path
$model = (Resolve-Path -LiteralPath $ModelPack).Path
$openvino = (Resolve-Path -LiteralPath $OpenVinoLibDir).Path
if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
  throw "Legal PDF binary does not exist: $binary"
}
if (-not (Test-Path -LiteralPath (Join-Path $model 'manifest.json') -PathType Leaf)) {
  throw "PPdoc model pack has no manifest.json: $model"
}
$runtimeFiles = @(
  'openvino_c.dll',
  'openvino.dll',
  'openvino_intel_cpu_plugin.dll',
  'openvino_onnx_frontend.dll',
  'openvino_ir_frontend.dll',
  'tbb12.dll'
)
if ($Gpu) { $runtimeFiles += 'openvino_intel_gpu_plugin.dll' }
foreach ($name in $runtimeFiles) {
  $source = Join-Path $openvino $name
  if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "OpenVINO runtime file does not exist: $source"
  }
}
if (Test-Path -LiteralPath $OutputDir) {
  if (Get-ChildItem -LiteralPath $OutputDir -Force | Select-Object -First 1) {
    throw "Output directory must be empty: $OutputDir"
  }
} else {
  New-Item -ItemType Directory -Path $OutputDir | Out-Null
}
$output = (Resolve-Path -LiteralPath $OutputDir).Path
$runtimeOutput = New-Item -ItemType Directory -Path (Join-Path $output 'runtime')
$modelOutput = New-Item -ItemType Directory -Path (Join-Path $output 'model')
Copy-Item -LiteralPath $binary -Destination (Join-Path $output 'legalpdf.exe')
foreach ($name in $runtimeFiles) {
  Copy-Item -LiteralPath (Join-Path $openvino $name) -Destination $runtimeOutput
}
Copy-Item -Path (Join-Path $model '*') -Destination $modelOutput -Recurse

$files = Get-ChildItem -LiteralPath $output -File -Recurse | Sort-Object FullName | ForEach-Object {
  $relative = $_.FullName.Substring($output.Length).TrimStart('\', '/')
  [pscustomobject]@{
    path = $relative.Replace('\', '/')
    bytes = $_.Length
    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}
[pscustomobject]@{
  format = 'legalpdf.ppdoc-windows-bundle/1'
  gpu = [bool]$Gpu
  files = @($files)
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $output 'bundle-manifest.json') -Encoding utf8

Write-Output $output
