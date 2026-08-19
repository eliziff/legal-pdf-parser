param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$ModelPack,
    [Parameter(Mandatory = $true)][string]$DatasetRoot,
    [Parameter(Mandatory = $true)][string]$ResultRoot,
    [ValidateSet('opencv', 'pillow')][string]$ImageBackend = 'opencv',
    [int[]]$Threads = @(1, 2, 4, 6, 8, 12, 0),
    [int[]]$BatchSizes = @(1),
    [int]$LimitPages = 10
)

$ErrorActionPreference = 'Stop'
$tool = Join-Path $SourceRoot 'tools\benchmark.py'
$annotations = Join-Path $DatasetRoot 'annotations\instance_val.json'
$images = Join-Path $DatasetRoot 'images'
$env:PYTHONPATH = Join-Path $SourceRoot 'src'
New-Item -ItemType Directory -Force -Path $ResultRoot | Out-Null
$status = Join-Path $ResultRoot 'status.json'
[IO.File]::WriteAllText($status, '{"phase":"running"}')

try {
    foreach ($batchSize in $BatchSizes) {
        foreach ($threadCount in $Threads) {
            $name = "teacher-fp32-rect_${ImageBackend}_t${threadCount}_b${batchSize}_${LimitPages}val.json"
            $output = Join-Path $ResultRoot $name
            & $Python $tool run `
                --model-pack $ModelPack `
                --annotations $annotations `
                --image-root $images `
                --output $output `
                --device cpu `
                --threads $threadCount `
                --batch-size $batchSize `
                --warmup-runs 2 `
                --limit-pages $LimitPages `
                --threshold 0.01 `
                --no-filter-overlap-boxes `
                --image-backend $ImageBackend `
                --no-score
            if ($LASTEXITCODE -ne 0) {
                throw "benchmark failed for threads=$threadCount batch=$batchSize"
            }
        }
    }
    [IO.File]::WriteAllText($status, '{"phase":"complete"}')
}
catch {
    $payload = @{ phase = 'failed'; error = $_.Exception.Message } | ConvertTo-Json -Compress
    [IO.File]::WriteAllText($status, $payload)
    throw
}
