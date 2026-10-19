from lxml import etree
import importlib.util
print('cairosvg',bool(importlib.util.find_spec('cairosvg')))
ns={'s':'http://www.w3.org/2000/svg'}
for name in ['02_eda_waveform_logmel','04_per_bin_zscore_normalization']:
 r=etree.parse('outputs/figures/'+name+'.svg')
 print(name,r.getroot().attrib)
 for x in r.xpath('//s:g',namespaces=ns):
  if not (x.get('id') or '').startswith('text_'):continue
  c=x.xpath('comment()')
  t=c[0].text if c else ''
  if any(y in t for y in ['Thời','Biên','Phổ','Phân bố']):
   print(x.get('id'),t,[(ch.tag,ch.get('transform')) for ch in x if isinstance(ch.tag,str)])
