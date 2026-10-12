param(
    [string]$ProjectRoot,
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$inferredProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if (-not $ProjectRoot) { $ProjectRoot = $inferredProjectRoot }
if (-not $Destination) {
    $Destination = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "..\..\autodl_upload"))
}
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Destination = [IO.Path]::GetFullPath($Destination)

$required = @(
    "configs\autodl\q2_v23_formal_ablation.yaml",
    "configs\autodl\q2_v23_formal_ablation_smoke.yaml",
    "configs\v2\BinarySup_5slice.yaml",
    "configs\v2\MeanTeacher_5slice.yaml",
    "configs\v2\CSD_Global_5slice.yaml",
    "scripts\autodl\29_run_v23_formal_ablation_all.sh",
    "scripts\analysis\analyze_v23_ablation_tkv.py",
    "pyproject.toml",
    "uv.lock"
)
foreach ($relative in $required) {
    $candidate = Join-Path $ProjectRoot $relative
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Required v2.3 file is missing: $candidate"
    }
}

New-Item -ItemType Directory -Path $Destination -Force | Out-Null
$archive = Join-Path $Destination "KidneyMRI-SpectralSSL_v23_formal_ablation_code.zip"
$hashPath = "$archive.sha256"
$manifestPath = Join-Path $Destination "KidneyMRI-SpectralSSL_v23_formal_ablation_manifest.txt"
foreach ($target in @($archive, $hashPath, $manifestPath)) {
    if (Test-Path -LiteralPath $target) {
        throw "Refusing to overwrite existing package output: $target"
    }
}

$tar = (Get-Command tar.exe -ErrorAction Stop).Source
$projectParent = Split-Path -Parent $ProjectRoot
$projectName = Split-Path -Leaf $ProjectRoot
& $tar -a -cf $archive `
    --exclude="$projectName/.git" `
    --exclude="$projectName/.venv" `
    --exclude="$projectName/.pytest_cache" `
    --exclude="$projectName/**/__pycache__" `
    --exclude="$projectName/.coverage" `
    --exclude="$projectName/build" `
    --exclude="$projectName/dist" `
    -C $projectParent $projectName
if ($LASTEXITCODE -ne 0) { throw "Creating the v2.3 code archive failed" }

$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([IO.Path]::GetFileName($archive))" | Set-Content -LiteralPath $hashPath -Encoding ascii
$commit = (& git -C $ProjectRoot rev-parse HEAD).Trim()
$status = (& git -C $ProjectRoot status --short) -join "`n"
@(
    "created_utc=$([DateTime]::UtcNow.ToString('o'))",
    "git_commit=$commit",
    "code_archive_sha256=$hash",
    "required_prior_archive=comparison_v22_csd_internal_compact_20260822T221119Z.tar.gz",
    "required_prior_archive_sha256=369f6788b5493c5adb4dd76cc8c657e683b65144ac3cd939f7ba9832fa2fec96",
    "formal_new_runs=45",
    "gpu=NVIDIA GeForce RTX 4090",
    "entrypoint=bash scripts/autodl/29_run_v23_formal_ablation_all.sh",
    "",
    "git_status_short:",
    $status
) | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Host $archive
Write-Host $hashPath
Write-Host $manifestPath
