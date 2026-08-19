param(
  [Parameter(Mandatory = $true)][string]$SdkDir,
  [Parameter(Mandatory = $true)][string]$OutputDir
)

$ErrorActionPreference = 'Stop'
$sdk = (Resolve-Path -LiteralPath $SdkDir).Path
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$output = (Resolve-Path -LiteralPath $OutputDir).Path

$files = @(
  'paddle\lib\paddle_inference.dll',
  'paddle\lib\phi.dll',
  'paddle\lib\common.dll',
  'third_party\install\onednn\lib\mkldnn.dll',
  'third_party\install\mklml\lib\libiomp5md.dll',
  'third_party\install\mklml\lib\mklml.dll'
)
foreach ($relative in $files) {
  Copy-Item -LiteralPath (Join-Path $sdk $relative) -Destination $output -Force
}

$builder = Join-Path $PSScriptRoot 'build_backend.cmd'
& $builder $sdk $output
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Get-ChildItem -LiteralPath $output -File -Filter '*.dll' |
  Sort-Object Name |
  ForEach-Object {
    $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    [pscustomobject]@{ name = $_.Name; bytes = $_.Length; sha256 = $hash.Hash.ToLowerInvariant() }
  } |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'runtime-files.json') -Encoding utf8
