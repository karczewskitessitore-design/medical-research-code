param(
    [string]$ProjectRoot,
    [string]$DataRoot,
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$inferredProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if (-not $ProjectRoot) { $ProjectRoot = $inferredProjectRoot }
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $ProjectRoot)
if (-not $DataRoot) { $DataRoot = Join-Path $workspaceRoot "data\T2KidneyMRI" }
if (-not $Destination) { $Destination = Join-Path $workspaceRoot "autodl_upload" }
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$DataRoot = [IO.Path]::GetFullPath($DataRoot)
$Destination = [IO.Path]::GetFullPath($Destination)

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "pyproject.toml") -PathType Leaf)) {
    throw "Project root is invalid: $ProjectRoot"
}
foreach ($name in @("CKD.zip", "Healthy_Control.zip", "subject_information.csv")) {
    $candidate = Join-Path $DataRoot "downloads\$name"
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Required verified download is missing: $candidate"
    }
}

New-Item -ItemType Directory -Path $Destination -Force | Out-Null
$codeArchive = Join-Path $Destination "KidneyMRI-SpectralSSL_code_fresh.zip"
$dataArchive = Join-Path $Destination "T2KidneyMRI_verified_downloads.zip"
foreach ($archive in @($codeArchive, $dataArchive)) {
    if (Test-Path -LiteralPath $archive) {
        throw "Refusing to overwrite an existing bundle: $archive"
    }
}

$tar = (Get-Command tar.exe -ErrorAction Stop).Source
$projectParent = Split-Path -Parent $ProjectRoot
$projectName = Split-Path -Leaf $ProjectRoot
& $tar -a -cf $codeArchive `
    --exclude="$projectName/.git" `
    --exclude="$projectName/.venv" `
    --exclude="$projectName/.pytest_cache" `
    --exclude="$projectName/**/__pycache__" `
    --exclude="$projectName/.coverage" `
    --exclude="$projectName/build" `
    --exclude="$projectName/dist" `
    -C $projectParent $projectName
if ($LASTEXITCODE -ne 0) { throw "Creating the code archive failed" }

$dataParent = Split-Path -Parent $DataRoot
$dataName = Split-Path -Leaf $DataRoot
& $tar -a -cf $dataArchive -C $dataParent "$dataName/downloads"
if ($LASTEXITCODE -ne 0) { throw "Creating the data archive failed" }

$codeHash = (Get-FileHash -LiteralPath $codeArchive -Algorithm SHA256).Hash.ToLowerInvariant()
$dataHash = (Get-FileHash -LiteralPath $dataArchive -Algorithm SHA256).Hash.ToLowerInvariant()
$commit = (& git -C $ProjectRoot rev-parse HEAD).Trim()
$status = (& git -C $ProjectRoot status --short) -join "`n"
$metadata = @(
    "created_utc=$([DateTime]::UtcNow.ToString('o'))"
    "git_commit=$commit"
    "code_archive_sha256=$codeHash"
    "data_archive_sha256=$dataHash"
    ""
    "git_status_short:"
    $status
)
$metadataPath = Join-Path $Destination "bundle_manifest.txt"
$metadata | Set-Content -LiteralPath $metadataPath -Encoding utf8

Write-Host "Created fresh AutoDL upload bundle:"
Write-Host "  $codeArchive"
Write-Host "  $dataArchive"
Write-Host "  $metadataPath"
Write-Host "No local virtual environment, old checkpoint, or experiment output is included."
