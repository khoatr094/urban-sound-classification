$ErrorActionPreference='Stop'
$w=New-Object -ComObject kwps.Application
$prior=$w.Documents.Count
if($prior -eq 0){$w.Visible=$false}
$w.DisplayAlerts=0
$d=$w.Documents.Open('D:\OD\Downloads\urban_sound\output\Maubaocao_KLTN_Xay_dung_he_thong_phan_loai_am_thanh_do_thi.docx',$false,$true)
$d.Fields.Update() | Out-Null
$d.ExportAsFixedFormat('D:\OD\Downloads\urban_sound\tmp\thesis_update\final.pdf',17)
$d.Close(0)
if($prior -eq 0){$w.Quit()}
Write-Output 'Exported updated report'
