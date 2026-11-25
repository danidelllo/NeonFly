param(
    [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$')][string]$Run,
    [Parameter(Mandatory)][int]$TrainerProcessId
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$log = Join-Path 'logs\ppo' $Run
$stopFile = Join-Path $log 'STOP'
$owner = Get-NetTCPConnection -LocalPort 47652 -State Listen
if ($owner.OwningProcess -ne $TrainerProcessId -or (Invoke-RestMethod 'http://127.0.0.1:47652/progress' -TimeoutSec 5).run -ne $Run) {
    throw 'The current trainer does not match the requested handoff.'
}
$trainer = Get-Process -Id $TrainerProcessId
$limit = (Get-Content (Join-Path $log 'config.json') -Raw | ConvertFrom-Json).config.updates
Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class TrainingPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
if (![TrainingPower]::SetThreadExecutionState([uint32]2147483649)) { throw 'Windows declined the keep-awake request.' }
try {
    Write-Output "Keeping Windows awake for $Run; waiting for trainer $TrainerProcessId to finish update $limit."
    while (!$trainer.WaitForExit(2000)) {
        if (Test-Path $stopFile) { return }
    }
    if (Test-Path $stopFile) { return }
    $last = Get-Content (Join-Path $log 'updates.jsonl') -Tail 1 | ConvertFrom-Json
    $eval = Get-Content (Join-Path $log 'evaluations.jsonl') -Tail 1 | ConvertFrom-Json
    if ($last.update -ne $limit -or $eval.update -ne $limit) { throw 'Trainer exited before its final checkpoint/evaluation. Not starting a second controller.' }
    & '.\.venv\Scripts\python.exe' -u -m python.viewer.serve --train-run $Run --resume --continuous --device cuda
    if ($LASTEXITCODE -ne 0 -or !(Test-Path $stopFile)) { throw 'Continuous trainer stopped unexpectedly; inspect the game and training logs.' }
} finally {
    [void][TrainingPower]::SetThreadExecutionState([uint32]2147483648)
}
