# Rebuild the final submission from the raw competition files and compare it with the submitted file.
# Run in Windows PowerShell:   powershell -ExecutionPolicy Bypass -File C:\Users\pc01\kaggle-midterm\reproduce.ps1
# Needs data\raw\ (see README), the virtual environment .venv, about 2 GB of free disk and 15-25 minutes.
# Derived tables go to data\processed_repro\, so the existing data\processed\ is not touched.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:KM_PROC = Join-Path $root "data\processed_repro"
New-Item -ItemType Directory -Force $env:KM_PROC | Out-Null
Set-Location (Join-Path $root "src")
$sw = [System.Diagnostics.Stopwatch]::StartNew()

& $py -X utf8 -W ignore build_interactions.py        # views x schedule -> interactions + schedule tables
& $py -X utf8 -W ignore build_all_overlaps.py         # overlaps with all airings (labelled and unlabelled)
& $py -X utf8 -W ignore build_folds2.py val12 val val2 val3 val5 test   # features of the five folds and of the test window
& $py -X utf8 -W ignore final3.py A5 reproduced_ensemble.csv           # ensemble of three re-rankers -> submissions\reproduced_ensemble.csv

"elapsed: {0:N1} min" -f $sw.Elapsed.TotalMinutes
Get-FileHash -Algorithm MD5 (Join-Path $root "submissions\reproduced_ensemble.csv"), (Join-Path $root "submissions\submission_v6_ensemble.csv") | Format-Table Hash, Path -AutoSize
$a = (Get-FileHash -Algorithm MD5 (Join-Path $root "submissions\reproduced_ensemble.csv")).Hash
$b = (Get-FileHash -Algorithm MD5 (Join-Path $root "submissions\submission_v6_ensemble.csv")).Hash
if ($a -eq $b) { "IDENTICAL: the rebuilt file equals the submitted file byte for byte" } else { "DIFFERENT: see README, section Reproducibility" }
