param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [string]$KValues = "5 10 15 20 25",

    [int]$MaxIter = 40,

    [int]$Seed = 42,

    [int]$NInit = 1,

    [ValidateSet("random", "kmedoids++", "hybrid")]
    [string]$InitStrategy = "random",

    [string]$OutputCsv = "results/kmeans_results.csv"
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path

try {
    $resolvedInputFile = (Resolve-Path $InputFile -ErrorAction Stop).Path
}
catch {
    throw "Input file not found: $InputFile"
}

if ([System.IO.Path]::IsPathRooted($OutputCsv)) {
    $resolvedOutputCsv = $OutputCsv
}
else {
    $resolvedOutputCsv = Join-Path $projectRoot $OutputCsv
}

Push-Location $projectRoot
try {
    Write-Host "[1/2] Running unit tests..." -ForegroundColor Cyan
    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "Unit tests failed. Fix tests before running experiment."
    }

    Write-Host "[2/2] Running clustering experiment..." -ForegroundColor Cyan
    python run_experiment.py --input-file "$resolvedInputFile" --k-values $KValues.Split(' ') --max-iter $MaxIter --seed $Seed --n-init $NInit --init-strategy $InitStrategy --output-csv "$resolvedOutputCsv"
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment run failed."
    }
}
finally {
    Pop-Location
}

Write-Host "Completed successfully." -ForegroundColor Green
