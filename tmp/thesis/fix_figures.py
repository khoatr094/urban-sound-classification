from lxml import etree
from pathlib import Path
import re
ns={'s':'http://www.w3.org/2000/svg'};S='{'+ns['s']+'}'
root=Path('tmp/thesis')
names=['Còi xe','Trẻ em vui chơi','Chó sủa','Động cơ chạy không tải','Tiếng súng','Còi báo động','Tiếng khoan']
r=etree.parse('outputs/figures/02_eda_waveform_logmel.svg')
for ax in r.xpath('//s:g[starts-with(@id,"axes_")]',namespaces=ns):
 n=int(ax.get('id').split('_')[-1]);row=(n-1)//2 if n<=14 else n-15
 # Add vertical breathing room without changing any data path or clip geometry.
 ax.set('transform',f'translate(0 {row*40})')
 for g in list(ax.xpath('.//s:g[starts-with(@id,"text_")]',namespaces=ns)):
  cs=g.xpath('comment()');t=cs[0].text.strip() if cs else ''
  if t.startswith('Biên độ sóng âm') or t.startswith('Phổ Log-Mel'):
   child=next(ch for ch in g if isinstance(ch.tag,str));tr=child.get('transform');xy=re.search(r'translate\(([-\d.]+) ([-\d.]+)\)',tr)
   if xy:
    y=float(xy.group(2));g.clear()
    x=240 if n%2==1 else 690
    tx=etree.SubElement(g,S+'text',x=str(x),y=str(y),fill='black',style='font-family: Times New Roman; font-size: 16px; font-weight: bold; text-anchor: middle')
    tx.text=names[row]+' — dạng sóng' if n%2==1 else names[row]+' — Log-Mel'
rt=r.getroot();rt.set('height','1297.752422pt');rt.set('viewBox','0 0 896.383884 1297.752422')
r.write(root/'waveform_clean.svg',xml_declaration=True,encoding='utf-8')
r=etree.parse('outputs/figures/04_per_bin_zscore_normalization.svg')
for g in r.xpath('//s:g[starts-with(@id,"text_")]',namespaces=ns):
 cs=g.xpath('comment()');t=cs[0].text.strip() if cs else ''
 if t.startswith('Phân bố năng lượng'):
  g.clear();x=225 if 'trước' in t else 610
  tx=etree.SubElement(g,S+'text',x=str(x),y='18',fill='black',style='font-family: Times New Roman; font-size: 15px; font-weight: bold; text-anchor: middle')
  tx.text='Trước chuẩn hóa' if 'trước' in t else 'Sau chuẩn hóa theo dải Mel'
r.write(root/'normalization_clean.svg',xml_declaration=True,encoding='utf-8')
r=etree.parse('outputs/figures/14_fps_stability_accuracy_comprehensive.svg')
ax=r.xpath('//s:g[starts-with(@id,"axes_")]',namespaces=ns)[3]
for g in ax.xpath('.//s:g[starts-with(@id,"text_")]',namespaces=ns):
 cs=g.xpath('comment()');t=cs[0].text.strip() if cs else ''
 if t in ['CNN 4 khối','ResCNN tự xây dựng','AST Pretrained']:
  title,x,y={'CNN 4 khối':('CNN4',980,578),'ResCNN tự xây dựng':('ResCNN',1000,642),'AST Pretrained':('AST',700,510)}[t]
  g.clear();tx=etree.SubElement(g,S+'text',x=str(x),y=str(y),fill='black',style='font-family: Times New Roman; font-size: 12px; font-weight: bold; text-anchor: middle');tx.text=title
r.write(root/'benchmark_clean.svg',xml_declaration=True,encoding='utf-8')
