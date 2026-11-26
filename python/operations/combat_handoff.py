"""Finish the fixed coverage budget, then install and launch the prepared combat run.

Not a general scheduler. Stops on interruption, unexpected source changes, an
existing competing controller, menu guards, or an already successful evaluation.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from python.communication.observe import observe, request

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'logs/ppo/flywire-coverage-006'
NEW = 'flywire-combat-007'
GAME = Path(r'C:\Program Files (x86)\Steam\steamapps\common\NEONSKY')


def read_rows(name):
    file = OLD/(name+'.jsonl')
    return [json.loads(line) for line in file.read_text().splitlines(keepends=True)
            if line.endswith('\n')] if file.exists() else []


def finished(updates, evaluations):
    return bool(updates and evaluations and updates[-1]['update'] == 128 and
                updates[-1]['episodes'] == 512 and evaluations[-1]['update'] == 128 and
                evaluations[-1]['episode'] == 512)


def check_manifest(manifest):
    for name, digest in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Prepared release changed: {name}')


def listen(port):
    with socket.socket() as sock:
        sock.settimeout(.5)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def powershell(text):
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', "$ErrorActionPreference='Stop'; "+text],
                            cwd=ROOT, text=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError('Handoff command failed: '+result.stderr.strip())


def quote(value):
    return "'"+str(value).replace("'", "''")+"'"


def wait_for(function, seconds, description):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        value = function()
        if value:
            return value
        time.sleep(.5)
    raise TimeoutError(description)


def game_status():
    try:
        return observe(False)[0]
    except (ConnectionError, OSError, EOFError):
        return None


def launch(manifest):
    check_manifest(manifest)
    parent = ROOT/'checkpoints/flywire-coverage-006/best.pt'
    import torch
    saved = torch.load(parent, map_location='cpu', weights_only=True)
    if saved['best_rank'][0] > 0:
        return dict(state='not_needed', reason='Coverage run achieved evaluated completions; preserved its model')
    if (ROOT/'checkpoints'/NEW).exists() or (ROOT/'logs/ppo'/NEW).exists():
        raise RuntimeError('Combat run already exists; refusing to overwrite it')
    wait_for(lambda: not listen(47652), 30, 'Old controller did not exit')
    # At this point the old trainer's final checkpoint/evaluation are complete.
    powershell("$game="+quote(GAME)+"; Get-CimInstance Win32_Process -Filter \"Name='NeonSky.exe'\" | ForEach-Object {"
               "if($_.ExecutablePath -ne (Join-Path $game 'NeonSky.exe')){throw 'Unexpected game process'}; "
               "$gameId=$_.ProcessId; Stop-Process -Id $gameId; Wait-Process -Id $gameId -Timeout 15 -ErrorAction SilentlyContinue; "
               "if(Get-Process -Id $gameId -ErrorAction SilentlyContinue){throw 'Game did not close'}}; "
               "$backup=New-Item -ItemType Directory -Force ('.deps/backups/before-combat-'+(Get-Date -Format 'yyyyMMdd-HHmmss')); "
               "Copy-Item -LiteralPath (Join-Path $game 'BepInEx/plugins/NeonFly.dll') -Destination $backup.FullName; "
               "Copy-Item -LiteralPath '.deps/combat-release/NeonFly.dll' -Destination (Join-Path $game 'BepInEx/plugins/NeonFly.dll'); "
               "Start-Sleep -Seconds 2; "
               "Start-Process 'C:/Program Files (x86)/Steam/steam.exe' -ArgumentList '-applaunch','4546400' -WindowStyle Hidden")
    status = wait_for(game_status, 90, 'New game bridge did not start')
    if status.get('bridgeVersion') != '0.9.0' or status.get('combatVersion') != 1:
        raise RuntimeError('Installed combat bridge did not report version 0.9/schema 1')
    # Follow ordinary menu routes only. NewGame refuses an existing save; no
    # save deletion or direct scene injection is used as a fallback.
    def enter_level():
        status = game_status()
        if not status:
            return False
        if status['scene'] == 'MainMenu':
            request(3)
        elif status['scene'] == 'LevelSelect':
            request(4)
        return status if status['scene'] == 'Level_1' else False
    wait_for(enter_level, 180, 'Normal menus/cutscenes did not reach Level_1')
    launch_dir = ROOT/'logs/launch'; launch_dir.mkdir(exist_ok=True)
    with (launch_dir/(NEW+'-out.txt')).open('wb') as out, (launch_dir/(NEW+'-err.txt')).open('wb') as err:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'python.viewer.serve', '--train-run', NEW,
            '--config', 'configs/combat_ppo.toml', '--env-config', 'configs/combat_ship.toml',
            '--warm-start', str(parent), '--device', 'cuda'], cwd=ROOT, stdout=out, stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW)
    (launch_dir/(NEW+'.pid')).write_text(str(process.pid))
    initial = ROOT/'checkpoints'/NEW/'initial.pt'
    def initialized():
        if process.poll() is not None:
            raise RuntimeError('Combat controller exited; inspect launch logs')
        return initial.exists()
    wait_for(initialized, 90, 'Combat controller did not initialize')
    started = torch.load(initial, map_location='cpu', weights_only=True)
    if started['progress'] != (0, 0, 0) or not all(torch.equal(value, started['head'][name]) for name, value in saved['head'].items()):
        raise RuntimeError('New controller did not inherit its selected parent')
    snapshot = ROOT/'logs/ppo'/NEW/'source'; snapshot.mkdir()
    import shutil
    for name in ('python', 'configs', 'unity-mod/src'):
        shutil.copytree(ROOT/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('AGENTS.md', 'docs/COMBAT_RUN.md'):
        shutil.copy2(ROOT/name, snapshot/Path(name).name)
    # Refresh only the spectator so its new status labels are served.
    powershell("$owner=Get-NetTCPConnection -LocalPort 47653 -State Listen -ErrorAction SilentlyContinue; "
               "if($owner){$p=Get-CimInstance Win32_Process -Filter ('ProcessId='+$owner.OwningProcess); "
               "if($p.CommandLine -notmatch 'python.viewer.serve.*--observe.*47653'){throw 'Unexpected observer'}; "
               "Stop-Process -Id $owner.OwningProcess}")
    with (launch_dir/(NEW+'-observer-out.txt')).open('wb') as out, (launch_dir/(NEW+'-observer-err.txt')).open('wb') as err:
        subprocess.Popen([sys.executable, '-u', '-m', 'python.viewer.serve', '--observe', '--port', '47653'],
                         cwd=ROOT, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    return dict(state='started', run=NEW, controller_pid=process.pid, parent=str(parent),
                parent_progress=saved['progress'], parent_rank=saved['best_rank'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('.deps/combat-release/manifest.json'))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(); manifest = json.loads(args.manifest.read_text())
    check_manifest(manifest)
    if args.check_only:
        print('Prepared combat release hashes verified; no process or game changes made.', flush=True)
        return
    result = ROOT/'logs/launch/combat-handoff-result.json'
    lock = ROOT/'.deps/combat-release/handoff.lock'
    with lock.open('x') as file:
        file.write(str(os.getpid()))
    import ctypes
    try:
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise RuntimeError('Windows refused handoff sleep inhibition')
        print('Waiting for coverage-006 episode 512 and final evaluation; settings unchanged.', flush=True)
        end = time.monotonic()+24*60*60
        while not finished(read_rows('updates'), read_rows('evaluations')):
            if (OLD/'STOP').exists():
                raise RuntimeError('Coverage run was intentionally stopped; cancelling handoff')
            if not listen(47652):
                if finished(read_rows('updates'), read_rows('evaluations')):
                    break
                raise RuntimeError('Coverage controller disappeared before its final evaluation')
            if time.monotonic() > end:
                raise TimeoutError('Coverage completion was not observed within 24 hours')
            with urllib.request.urlopen('http://127.0.0.1:47652/progress', timeout=4) as response:
                if json.load(response).get('run') != 'flywire-coverage-006':
                    raise RuntimeError('Another run replaced the coverage controller')
            with urllib.request.urlopen('http://127.0.0.1:47652/control', timeout=4) as response:
                control = json.load(response)
            if control.get('error') or control['state'] == 'stopped':
                if finished(read_rows('updates'), read_rows('evaluations')):
                    break
                raise RuntimeError('Coverage worker failed before completion')
            time.sleep(2)
        value = launch(manifest)
        result.write_text(json.dumps(value, indent=2)); print(json.dumps(value), flush=True)
    except Exception as error:
        result.write_text(json.dumps(dict(state='failed', reason=str(error)), indent=2))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink()


if __name__ == '__main__':
    main()
