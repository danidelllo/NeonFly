param([string]$Game = 'C:\Program Files (x86)\Steam\steamapps\common\NEONSKY')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$deps = Join-Path $root '.deps'
$zip = Join-Path $deps 'BepInEx-5.4.23.6.zip'
$loader = Join-Path $deps 'loader'
$output = Join-Path $root 'unity-mod\build'
$managed = Join-Path $Game 'NeonSky_Data\Managed'
$compiler = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
New-Item -ItemType Directory -Force $deps,$output | Out-Null
if (!(Test-Path $zip)) {
    Invoke-WebRequest -UseBasicParsing 'https://github.com/BepInEx/BepInEx/releases/download/v5.4.23.6/BepInEx_win_x64_5.4.23.6.zip' -OutFile $zip
}
if ((Get-FileHash $zip).Hash -ne '6c57a714a892efb4e9e1fef63d608a3391f30ec984031c465404e4ee381a0755') {
    throw 'BepInEx archive digest mismatch.'
}
Expand-Archive $zip $loader -Force
$refs = @('mscorlib','netstandard','System','System.Core','Assembly-CSharp','UnityEngine','UnityEngine.InputLegacyModule','UnityEngine.Physics2DModule','UnityEngine.UI','UnityEngine.UIModule','UnityEngine.CoreModule','UnityEngine.ScreenCaptureModule','UnityEngine.ImageConversionModule','UnityEngine.JSONSerializeModule') | ForEach-Object { "/reference:$managed\$_.dll" }
& $compiler /nologo /noconfig /nostdlib+ /target:library /optimize+ /warnaserror+ "/out:$output\NeonFly.dll" "/reference:$loader\BepInEx\core\BepInEx.dll" "/reference:$loader\BepInEx\core\0Harmony.dll" $refs "$root\unity-mod\src\Bridge.cs" "$root\unity-mod\src\GameAdapter.cs" "$root\unity-mod\src\RangeSensor.cs" "$root\unity-mod\src\Combat.cs" "$root\unity-mod\src\CombatLedger.cs"
if ($LASTEXITCODE -ne 0) { throw 'Bridge compilation failed.' }
Write-Output "$output\NeonFly.dll"
