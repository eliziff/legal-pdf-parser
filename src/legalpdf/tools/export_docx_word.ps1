[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $InputPath,

    [Parameter(Mandatory = $true)]
    [string] $OutputPath,

    [Parameter(Mandatory = $true)]
    [ValidateSet('native', 'print')]
    [string] $Profile
)

$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open(
        [System.IO.Path]::GetFullPath($InputPath),
        $false,
        $true
    )
    $output = [System.IO.Path]::GetFullPath($OutputPath)
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($output)) | Out-Null
    $optimizeFor = if ($Profile -eq 'native') { 1 } else { 0 }
    $documentStructureTags = $Profile -eq 'native'
    $document.ExportAsFixedFormat(
        $output,
        17,
        $false,
        $optimizeFor,
        0,
        1,
        1,
        0,
        $true,
        $true,
        0,
        $documentStructureTags,
        $true,
        $false
    )
    Write-Output ('Microsoft Word ' + $word.Version)
}
finally {
    if ($null -ne $document) {
        $document.Close(0)
        [void] [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void] [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
