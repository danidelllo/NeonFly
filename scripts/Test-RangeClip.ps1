$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$compiler = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
$exe = Join-Path $root '.deps\RangeClipTests.exe'
& $compiler /nologo /target:exe "/out:$exe" (Join-Path $root 'unity-mod\tests\RangeClipTests.cs')
if ($LASTEXITCODE) { throw 'Range clipping test compilation failed' }
& $exe (Join-Path $root 'unity-mod\build\NeonFly.dll') 'C:\Program Files (x86)\Steam\steamapps\common\NEONSKY\NeonSky_Data\Managed'
if ($LASTEXITCODE) { throw 'Range clipping tests failed' }
