$ErrorActionPreference='Stop'
$w=New-Object -ComObject kwps.Application
$w.Visible=$false
$w.DisplayAlerts=0
$d=$w.Documents.Open('D:\OD\KL\Mau bao cao KLTN_DS.docx',$false,$true)
$d.ExportAsFixedFormat('D:\OD\Downloads\urban_sound\tmp\thesis\template.pdf',17)
$d.Close(0)
$w.Quit()
Write-Output 'Exported template'
