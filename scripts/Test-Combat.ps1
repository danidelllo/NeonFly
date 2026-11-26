param([string]$Game = 'C:\Program Files (x86)\Steam\steamapps\common\NEONSKY')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$compiler = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
$exe = Join-Path $root '.deps\CombatTests.exe'
$core = Join-Path $root '.deps\loader\BepInEx\core'
& $compiler /nologo /target:exe "/out:$exe" "/reference:$core\0Harmony.dll" (Join-Path $root 'unity-mod\tests\CombatTests.cs')
if ($LASTEXITCODE) { throw 'Combat test compilation failed' }
& $exe (Join-Path $root 'unity-mod\build\NeonFly.dll') "$Game\NeonSky_Data\Managed" $core
if ($LASTEXITCODE) { throw 'Combat tests failed' }
