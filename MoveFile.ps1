Import-CSV -Path C:\Users\Devin\Documents\GitHub\ASL_Data\ASL_Data.csv -Header destination, filename | Foreach-Object { 
Move-Item $_.filename -Destination $_.destination
}

pause

