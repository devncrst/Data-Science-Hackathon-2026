# PowerShell twin of run_all.sh
$ErrorActionPreference = "Stop"
function Invoke-H4d {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs)
    & python -m h4d.cli @CliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "h4d.cli failed ($LASTEXITCODE): $($CliArgs -join ' ')"
    }
}
if (-not $env:H4D_DATA_DIR) {
    throw "set H4D_DATA_DIR to the folder with the two parquet files"
}
if (-not $env:H4D_OUT_DIR) {
    $env:H4D_OUT_DIR = (Resolve-Path ".\outputs" -ErrorAction SilentlyContinue)
    if (-not $env:H4D_OUT_DIR) {
        New-Item -ItemType Directory -Path ".\outputs" | Out-Null
        $env:H4D_OUT_DIR = (Resolve-Path ".\outputs").Path
    }
}

Invoke-H4d --force preprocess
Invoke-H4d --force stage1
Invoke-H4d --force zeroshot
Invoke-H4d sweep
Invoke-H4d sweep --rng-mode notebook
foreach ($n in 5, 25, 50, 100, 200) {
    Invoke-H4d adapt --n $n
}
Invoke-H4d export
Invoke-H4d dashboard
Write-Host "DONE → $($env:H4D_OUT_DIR)"
