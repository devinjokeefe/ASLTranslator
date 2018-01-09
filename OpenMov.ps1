$urllist = Get-Content C:\Users\Devin\Documents\GitHub\ASL_Data\Data_URL.txt
foreach($url in $urllist)
{
    $proc = start-process "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" -argumentlist $url -PassThru
    start-sleep -seconds 1
} 
