param([ValidateSet('cuda','cpu')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root '.venv\Scripts\python.exe'
$edge = Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'
$url = 'http://127.0.0.1:47652'
if (!(Test-Path $edge)) { throw "Microsoft Edge not found: $edge" }
Push-Location $root
try {
    if (!(Test-Path '.deps\flywire783\manifest.json')) {
        & $python -m python.viewer.prepare
        if ($LASTEXITCODE) { throw 'Could not prepare pinned FlyWire data.' }
    }
    $listener = Get-NetTCPConnection -LocalPort 47652 -State Listen -ErrorAction SilentlyContinue
    $existingProducer = [bool]$listener
    if (!$listener) {
        $dir = New-Item -ItemType Directory -Force 'logs\viewer'
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $p = Start-Process $python -ArgumentList @('-u','-m','python.viewer.serve','--device',$Device) -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput "$dir\$stamp-out.txt" -RedirectStandardError "$dir\$stamp-err.txt"
        $limit = (Get-Date).AddSeconds(30)
        do {
            if ($p.HasExited) { throw "Observatory exited; see $dir\$stamp-err.txt" }
            Start-Sleep -Milliseconds 250
            $listener = Get-NetTCPConnection -LocalPort 47652 -State Listen -ErrorAction SilentlyContinue
        } while (!$listener -and (Get-Date) -lt $limit)
        if (!$listener) { throw "Observatory did not listen within 30s; see $dir" }
    }
    $meta = Invoke-RestMethod "$url/meta" -TimeoutSec 5
    if ($meta.release -ne 'FAFB v783' -or $meta.nodes -ne 139255) { throw 'Unexpected service on port 47652' }
    if ($existingProducer) {
        # Updated UI can run beside an existing trainer without restarting it.
        $url = 'http://127.0.0.1:47653'
        $observer = Get-NetTCPConnection -LocalPort 47653 -State Listen -ErrorAction SilentlyContinue
        if (!$observer) {
            $dir = New-Item -ItemType Directory -Force 'logs\viewer'
            $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
            $p = Start-Process $python -ArgumentList @('-u','-m','python.viewer.serve','--observe','--port','47653') -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput "$dir\$stamp-observe-out.txt" -RedirectStandardError "$dir\$stamp-observe-err.txt"
            $limit = (Get-Date).AddSeconds(30)
            do {
                if ($p.HasExited) { throw "Observer exited; see $dir\$stamp-observe-err.txt" }
                Start-Sleep -Milliseconds 250
                $observer = Get-NetTCPConnection -LocalPort 47653 -State Listen -ErrorAction SilentlyContinue
            } while (!$observer -and (Get-Date) -lt $limit)
            if (!$observer) { throw 'Read-only Observatory did not start within 30s' }
        }
        $observerMeta = Invoke-RestMethod "$url/meta" -TimeoutSec 5
        if ($observerMeta.release -ne 'FAFB v783') { throw 'Unexpected observer on port 47653' }
    }
    $profile = Join-Path $root '.deps\viewer-edge'
    Start-Process $edge -ArgumentList @("--user-data-dir=`"$profile`"",'--no-first-run','--no-default-browser-check',"--app=$url/",'--window-size=1260,800')
} finally {
    Pop-Location
}
