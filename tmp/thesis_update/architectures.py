from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
OUT=Path(__file__).resolve().parent
def f(n=42,b=False):return ImageFont.truetype('C:/Windows/Fonts/'+('timesbd.ttf' if b else 'times.ttf'),n)
def start(h,title):
 im=Image.new('RGB',(2200,h),'white');dr=ImageDraw.Draw(im);dr.text((1100,60),title,font=f(58,True),anchor='mm',fill='black');return im,dr
def box(dr,rect,title,detail='',fill='#eef3f8',sz=42):
 dr.rounded_rectangle(rect,radius=15,fill=fill,outline='#283748',width=3)
 x0,y0,x1,y1=rect;cx=(x0+x1)/2;lines=title.split('\n')
 allh=len(lines)*52+(len(detail.split('\n'))*45+18 if detail else 0)
 y=(y0+y1-allh)/2+26
 for s in lines:dr.text((cx,y),s,font=f(sz,True),anchor='mm',fill='black');y+=52
 if detail:
  y+=12
  for s in detail.split('\n'):dr.text((cx,y),s,font=f(36),anchor='mm',fill='black');y+=45
def arrow(dr,a,b):
 dr.line((a,b),fill='#263442',width=5)
 import math
 t=math.atan2(b[1]-a[1],b[0]-a[0]);L=19
 dr.polygon([b,(b[0]-L*math.cos(t-.5),b[1]-L*math.sin(t-.5)),(b[0]-L*math.cos(t+.5),b[1]-L*math.sin(t+.5))],fill='#263442')
def note(dr,xy,t):dr.text(xy,t,font=f(35),anchor='mm',fill='black')
im,dr=start(1440,'CNN4  Cấu trúc lớp và kích thước tensor')
labels=[('Log-Mel','1 × 64 × 251'),('Khối 1','32 × 32 × 125'),('Khối 2','64 × 16 × 62'),('Khối 3','128 × 8 × 31'),('Khối 4','256 × 4 × 15')]
for i,(a,b) in enumerate(labels):
 x=60+i*435;box(dr,(x,180,x+350,405),a,b)
 if i<4:arrow(dr,(x+350,292),(x+425,292))
note(dr,(1100,470),'Kích thước theo thứ tự kênh × tần số × thời gian; pooling lấy phần nguyên.')
box(dr,(120,590,2080,785),'Chi tiết một khối CNN','Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU → MaxPool 2×2 → Dropout2d',sz=46)
note(dr,(1100,850),'Số kênh 32 → 64 → 128 → 256; dropout 0,10 → 0,15 → 0,20 → 0,25.')
for rect,title,detail in [((100,990,600,1205),'GAP','256 đặc trưng'),((825,990,1375,1205),'Linear 256 → 128','ReLU + Dropout 0,35'),((1600,990,2100,1205),'Linear 128 → 7','Logits → softmax')]:box(dr,rect,title,detail)
arrow(dr,(600,1100),(815,1100));arrow(dr,(1375,1100),(1590,1100))
note(dr,(1100,1310),'Mỗi tích chập có padding 1; stride 1. GAP là lấy trung bình toàn cục.')
im.save(OUT/'cnn4_detail.png')

im,dr=start(1500,'ResCNN  Các tầng phần dư và tổng hợp đặc trưng')
labels=[('Stem','32 × 64 × 251'),('Khối 1 + SE','64 × 32 × 126'),('Khối 2 + SE','128 × 16 × 63'),('Khối 3 + SE','256 × 8 × 32'),('Khối 4 + SE','256 × 4 × 16')]
for i,(a,b) in enumerate(labels):
 x=60+i*435;box(dr,(x,180,x+350,420),a,b,sz=38)
 if i<4:arrow(dr,(x+350,300),(x+425,300))
note(dr,(1100,485),'Stem: Conv 3×3 + BN + GELU. Mỗi khối phần dư giảm kích thước với stride 2.')
arrow(dr,(1975,420),(1975,570));dr.line((1975,570,540,570),fill='#263442',width=5)
arrow(dr,(540,570),(540,660));arrow(dr,(1660,570),(1660,660))
box(dr,(180,665,900,865),'Global average pooling','Vector 256 chiều')
box(dr,(1300,665,2020,865),'Global max pooling','Vector 256 chiều')
arrow(dr,(540,865),(540,940));arrow(dr,(1660,865),(1660,940));dr.line((540,940,1660,940),fill='#263442',width=5);arrow(dr,(1100,940),(1100,1010))
box(dr,(650,1015,1550,1175),'Ghép hai vector','512 chiều → LayerNorm → Dropout 0,35')
arrow(dr,(1100,1175),(1100,1240));box(dr,(260,1245,1940,1410),'Tầng phân loại','Linear 512 → 128 → GELU → Dropout 0,25 → Linear 128 → 7')
im.save(OUT/'rescnn_detail.png')

im,dr=start(2050,'Khối phần dư kết hợp chú ý kênh SE')
box(dr,(700,145,1500,290),'Đặc trưng đầu vào')
arrow(dr,(1100,290),(1100,355));dr.line((350,320,1100,320),fill='#263442',width=5)
box(dr,(700,360,1500,520),'Conv 3×3 + BN + GELU','Stride 2 ở khối giảm kích thước')
arrow(dr,(1100,520),(1100,585));box(dr,(700,590,1500,735),'Conv 3×3 + BN','Stride 1')
arrow(dr,(1100,735),(1100,795));box(dr,(700,800,1500,945),'Chú ý kênh SE','Nhân trọng số theo từng kênh')
arrow(dr,(1100,945),(1100,1000));box(dr,(700,1005,1500,1135),'Dropout2d')
arrow(dr,(1100,1135),(1100,1225));dr.ellipse((1045,1225,1155,1335),fill='white',outline='#263442',width=4);dr.text((1100,1275),'+',font=f(68,True),anchor='mm',fill='black')
arrow(dr,(350,320),(350,665));box(dr,(95,670,605,870),'Nhánh tắt','Identity hoặc\nConv 1×1 + BN',sz=40)
dr.line((350,870,350,1280),fill='#263442',width=5);arrow(dr,(350,1280),(1045,1280))
arrow(dr,(1100,1335),(1100,1395));box(dr,(700,1400,1500,1530),'GELU và đầu ra')
box(dr,(80,1640,2120,1955),'Bên trong SE','Global average pooling → Conv 1×1 giảm kênh → ReLU\n→ Conv 1×1 khôi phục số kênh → Sigmoid → trọng số kênh',sz=44)
im.save(OUT/'resblock_detail.png')

im,dr=start(1760,'AST  Biểu diễn patch và Transformer encoder')
for i,(a,b) in enumerate([
 ('Log-Mel','1.024 × 128'),('Patch 16×16','Stride 10 × 10'),('1.212 patch','Chiếu mỗi patch\nthành 768 chiều'),('Token đầu vào','Thêm CLS, DIST\nvà vị trí')]):
 x=80+i*545;box(dr,(x,175,x+410,420),a,b,sz=42)
 if i<3:arrow(dr,(x+410,295),(x+530,295))
note(dr,(1100,480),'12 vị trí theo tần số × 101 vị trí theo thời gian = 1.212 patch; thêm hai token đặc biệt.')
box(dr,(180,590,2020,840),'Chuỗi 1.214 × 768 → 12 Transformer encoder','Mỗi encoder: 12 đầu attention; kích thước tầng ẩn MLP là 3.072.',sz=47)
arrow(dr,(1100,840),(1100,930));box(dr,(420,935,1780,1110),'Biểu diễn toàn cục 768 chiều','LayerNorm → trung bình hai token CLS và DIST')
arrow(dr,(1100,1110),(1100,1190));box(dr,(500,1195,1700,1340),'Đầu phân loại 7 lớp','LayerNorm → Linear 768 → 7 → logits')
box(dr,(100,1440,2100,1670),'Một Transformer encoder','LayerNorm → Multi-head self-attention → cộng nhánh tắt\n→ LayerNorm → MLP 768 → 3.072 → 768 → cộng nhánh tắt',sz=44)
im.save(OUT/'ast_detail.png')
