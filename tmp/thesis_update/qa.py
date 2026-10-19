from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image,ImageDraw
from lxml import etree
import json,re,sys,zipfile
root=Path(__file__).resolve().parent
pdf=PdfReader(root/'final.pdf');texts=[(p.extract_text() or '').encode('utf-16','surrogatepass').decode('utf-16','replace') for p in pdf.pages]
def norm(s):return re.sub(r'[^\w]','',s).casefold()
nt=[norm(s) for s in texts]
first=next(i for i,s in enumerate(nt) if 'chương1giớithiệuvềbàitoán' in s and 'âmthanhlàmộtnguồnthôngtinquenthuộc' in s)
loc=json.loads((root/'locators.json').read_text(encoding='utf8'));cache={}
for title,key,*rest in loc['outline']+loc['figs']+loc['tables']:
 matches=[i for i in range(first,len(nt)) if norm(title) in nt[i]]
 if matches:cache[key]=str(matches[0]-first+1)
 else:print('MISSING',key,title)
cache.update(refs='a',appendix='',journal='',tabA1='',tabB1='')
(root/'pagecache.json').write_text(json.dumps(cache,ensure_ascii=False,indent=2),encoding='utf8')
(root/'rendered_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+s for i,s in enumerate(texts)),encoding='utf8')
doc=root.parents[1]/'output/Maubaocao_KLTN_Xay_dung_he_thong_phan_loai_am_thanh_do_thi.docx'
z=zipfile.ZipFile(doc);r=etree.fromstring(z.read('word/document.xml'))
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
names=set(r.xpath('//w:bookmarkStart/@w:name',namespaces=ns));anchors=r.xpath('//w:hyperlink/@w:anchor',namespaces=ns)
broken=[a for a in anchors if a not in names]
mathtext=''.join(r.xpath('//m:oMath//m:t/text()',namespaces=ns))
assert not broken,broken
assert '/' not in mathtext
assert len(r.xpath('//m:f',namespaces=ns))>=8
assert len(r.xpath('//m:oMath',namespaces=ns))==13
rels=etree.fromstring(z.read('word/_rels/document.xml.rels'));external=[x.get('Target') for x in rels if x.get('Type','').endswith('/hyperlink') and x.get('TargetMode')=='External']
assert len(external)>=9
old=zipfile.ZipFile(root/'source_snapshot.docx');assert z.read('word/styles.xml')==old.read('word/styles.xml')
print('pages',len(pdf.pages),'main starts',first+1,'native equations',13,'stacked fractions',len(r.xpath('//m:f',namespaces=ns)),'internal links',len(anchors),'web links',len(external),'broken links',broken)
print('PDF hyperlinks',sum(len(p.get('/Annots',[])) for p in pdf.pages))
if '--images' in sys.argv:
 dest=root/'render';dest.mkdir(exist_ok=True)
 paths=convert_from_path(root/'final.pdf',dpi=110,poppler_path=r'C:/Users/panh3/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin',output_folder=str(dest),fmt='png',paths_only=True)
 (root/'pages.json').write_text(json.dumps(paths),encoding='utf8')
 for k in range(0,len(paths),12):
  sheet=Image.new('RGB',(1600,1725),'#ddd');dr=ImageDraw.Draw(sheet)
  for j,path in enumerate(paths[k:k+12]):
   im=Image.open(path);im.thumbnail((390,550));sheet.paste(im,((j%4)*400,(j//4)*575));dr.text(((j%4)*400+8,(j//4)*575+550),str(k+j+1),fill='black')
  sheet.save(dest/f'contact_{k//12+1}.jpg')
