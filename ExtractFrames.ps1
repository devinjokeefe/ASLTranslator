$folders = Get-ChildItem -dir

foreach ($directory in $folders) {
  cd $directory
  $videos = Get-ChildItem
    foreach ($mov in $videos) {
      $name = Split-Path $mov -leaf
      $mov
      "MOVIE PATH"
      $name = $name -replace '[.mov]',''
      $total_duration = ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $mov
      $fps = 50 / $total_duration
      Write-Output $total_duration
      ffmpeg -i "$mov" -vf fps=$fps "${name}_%02d.png"
    }
  cd C:\Users\Devin\Documents\GitHub\ASL_Data\Train_Data	
}
pause
