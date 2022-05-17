# $financeIndicators = Import-Csv('./CSVinfo/')

$techIndicators = Get-ChildItem "./CSVinfo"
$percentageChangeAI = @{}

# foreach ($item in $techIndicators) {
  # $files += $item
  $fiObject = Import-Csv("CSVinfo/$techIndicators[2]")
  # }
  $CSV_headers = $fiObject[0].psobject.properties.name
  $fiObject
  $fiObject.Length
  
  
  $percentageChangeAI | Add-Member -MemberType ScriptMethod -Name "Percentage Change" -Value $fiObject
  
  $percentageChangeAI


# for ($x = 0; $x -lt $fiObject.Length;$x++) {
#   switch ($fiObject."Daily Percent Change") {
#     {$fiObject."Daily Percent Change"[$x] -lt -0.01} {
#       $percentageChangeAI | Add-Member -MemberType NoteProperty -Name "Percent Change" -Value 0 -Force
#     }
#     {$fiObject."Daily Percent Change"[$x] -lt -0.005} {
#       $percentageChangeAI | Add-Member -MemberType NoteProperty -Name "Percent Change" -Value 0.2 -Force
#     }
#     {$fiObject."Daily Percent Change"[$x] -lt 0} {
#       $percentageChangeAI | Add-Member -MemberType NoteProperty -Name "Percent Change" -Value 0.4 -Force
#     }
#     {$fiObject."Daily Percent Change"[$x] -gt 0} {
#       $percentageChangeAI | Add-Member -MemberType NoteProperty -Name "Percent Change" -Value 0.6 -Force
#     }
#     {$fiObject."Daily Percent Change"[$x] -gt 0.005} {
#       $percentageChangeAI | Add-Member -MemberType NoteProperty -Name "Percent Change" -Value 0.8 -Force
#     }
#     {$fiObject."Daily Percent Change"[$x] -gt 0.01} {
#       $percentageChangeAI | Add-Member -MemberType NoteProperty -Name "Percent Change" -Value 1 -Force
#     }
#     Default {
#       throw
#     }
#   }
# }

$percentageChangeAI
