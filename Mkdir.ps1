$list = Get-Content -Path "C:\Users\Devin\Documents\GitHub\ASL_Data\Data_Folders.txt"

foreach ($line in $list) {
  New-Item -ItemType directory -Path C:\Users\Devin\Documents\GitHub\ASL_Data\Test_Data -Name $line
  New-Item -ItemType directory -Path C:\Users\Devin\Documents\GitHub\ASL_Data\Train_Data -Name $line 
}

pause
