$exePath  = "C:\Users\abdul\Downloads\Practice\dist\STAR Analyzer\STAR Analyzer.exe"
$lnkPath  = [System.IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'STAR Analyzer.lnk')
$ws = New-Object -ComObject WScript.Shell
$s  = $ws.CreateShortcut($lnkPath)
$s.TargetPath       = $exePath
$s.WorkingDirectory = [System.IO.Path]::GetDirectoryName($exePath)
$s.IconLocation     = $exePath
$s.Description      = "STAR Analyzer"
$s.Save()
Write-Host "Shortcut created at: $lnkPath"
Write-Host "Target: $($s.TargetPath)"
