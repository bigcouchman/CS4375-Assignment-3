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

if (-not (Test-Path $InputFile)) {
    throw "Input file not found: $InputFile"
}

Write-Host "[1/2] Running unit tests..." -ForegroundColor Cyan
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) {
    throw "Unit tests failed. Fix tests before running experiment."
}

Write-Host "[2/2] Running clustering experiment..." -ForegroundColor Cyan
python run_experiment.py --input-file $InputFile --k-values $KValues.Split(' ') --max-iter $MaxIter --seed $Seed --n-init $NInit --init-strategy $InitStrategy --output-csv $OutputCsv
if ($LASTEXITCODE -ne 0) {
    throw "Experiment run failed."
}

Write-Host "Completed successfully." -ForegroundColor Green
