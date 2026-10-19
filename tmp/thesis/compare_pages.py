from PIL import Image,ImageChops
from pathlib import Path
import json
r=Path(__file__).resolve().parent
old=json.loads((r/'previous_pages.json').read_text());new=json.loads((r/'pages.json').read_text())
same=[];changed=[]
for j,path in enumerate(new):
 old_i=j if j<24 else j-1 if j>=30 else None
 if old_i is None:changed.append(j+1);continue
 a=Image.open(old[old_i]).convert('RGB');b=Image.open(path).convert('RGB')
 box=(0,110,a.width,1160) if j>=14 else (0,0,a.width,a.height)
 a=a.crop(box);b=b.crop(box)
 if a.size==b.size and ImageChops.difference(a,b).getbbox() is None:same.append(j+1)
 else:changed.append(j+1)
print('Previously inspected identical page bodies:',same)
print('Review current page bodies:',changed)
(r/'pages_to_inspect.json').write_text(json.dumps(changed))
