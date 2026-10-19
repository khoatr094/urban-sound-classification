import zipfile,xml.etree.ElementTree as E,json
p=r'D:/OD/KL/Mau bao cao KLTN_DS.docx'
z=zipfile.ZipFile(p); ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
r=E.fromstring(z.read('word/document.xml'))
with open('tmp/thesis/template.txt','w',encoding='utf8') as f:
 for i,e in enumerate(r.find('w:body',ns)):
  t=''.join(e.itertext()) if False else ''.join(x.text or '' for x in e.findall('.//w:t',ns))
  f.write(f'{i}: {t}\n')
  if e.tag.endswith('sectPr'): f.write(E.tostring(e,encoding='unicode')+'\n')
open('tmp/thesis/styles.xml','wb').write(z.read('word/styles.xml'))
print(open('tmp/thesis/template.txt',encoding='utf8').read())
