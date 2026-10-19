$ErrorActionPreference='Stop'
$w=New-Object -ComObject kwps.Application
$w.Visible=$false
$w.DisplayAlerts=0
$d=$w.Documents.Open('D:\OD\Downloads\urban_sound\output\Maubaocao_KLTN_Phan_loai_am_thanh_do_thi.docx',$false,$true)
$d.ExportAsFixedFormat('D:\OD\Downloads\urban_sound\tmp\thesis\final.pdf',17)
$d.Close(0)
$w.Quit()
Write-Output 'Exported report'
