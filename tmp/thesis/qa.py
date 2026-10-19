from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image,ImageDraw
import json,re,sys
root=Path(__file__).resolve().parent
reader=PdfReader(root/'final.pdf')
texts=[p.extract_text() or '' for p in reader.pages]
def norm(s):return re.sub(r'[^\w]','',s).casefold()
nt=[norm(t) for t in texts]
loc=json.loads((root/'locators.json').read_text(encoding='utf8'))
first=next(i for i,s in enumerate(nt) if i>10 and 'chương1giớithiệuvềbàitoán' in s)
ref=next(i for i,s in enumerate(nt) if i>first and 'tàiliệuthamkhảotiếnganh1gong' in s)
cache={}
for title,key,*rest in loc['outline']+loc['figs']+loc['tables']:
 matches=[i for i in range(first,len(nt)) if norm(title) in nt[i]]
 if matches:cache[key]=str(matches[0]-first+1)
 else:print('MISSING',key,title)
cache['refs']='a';cache['appendix']='';cache['journal']='';cache['tabA1']='';cache['tabB1']=''
(root/'pagecache.json').write_text(json.dumps(cache,ensure_ascii=False,indent=2),encoding='utf8')
(root/'rendered_text.txt').write_text('\n\n'.join('PAGE '+str(i+1)+'\n'+s for i,s in enumerate(texts)),encoding='utf8')
print('PAGES',len(texts),'main',first+1,'refs',ref+1,'cache',len(cache))
print('SPARSE',[(i+1,len(s)) for i,s in enumerate(texts) if len(s)<600])
for i,s in enumerate(texts):
 if 'Error!' in s or 'HƯỚNG DẪN VÀ MẪU' in s:print('BAD',i+1)
if '--images' in sys.argv:
 dest=root/'qa2';dest.mkdir(exist_ok=True)
 a=convert_from_path(root/'final.pdf',dpi=110,poppler_path=r'C:/Users/panh3/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin',output_folder=str(dest),fmt='png',paths_only=True)
 (root/'pages.json').write_text(json.dumps(a),encoding='utf8')
 for k in range(0,len(a),12):
  out=Image.new('RGB',(1600,1725),'#ddd'); dr=ImageDraw.Draw(out)
  for j,f in enumerate(a[k:k+12]):
   im=Image.open(f);im.thumbnail((390,550));out.paste(im,((j%4)*400,(j//4)*575));dr.text(((j%4)*400+10,(j//4)*575+550),str(k+j+1),fill='black')
  out.save(dest/f'contact_{k//12+1}.jpg')
