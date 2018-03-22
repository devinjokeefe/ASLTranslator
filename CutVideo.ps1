$videos = Get-ChildItem C:\Users\Devin\Documents\GitHub\ASL_Data\Vids\*.mov
foreach ($mov in $videos)
	{
	  $name = Split-Path $mov -leaf
	  $filename = "C:\Users\Devin\Documents\GitHub\ASL_Data\Videos\" + $name
	  $total_duration = ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $mov
	  $desired_duration = $total_duration - 4
	  ffmpeg -i $mov -c copy -ss 00:00:02 -t $desired_duration $filename
  	}
pause
