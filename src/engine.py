# -*- coding: utf-8 -*-
import os, sys, json, math, time, random, hashlib, gc, shutil, platform, traceback, warnings, threading
from pathlib import Path
from contextlib import nullcontext
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score)
from IPython.display import display, Markdown, Audio, FileLink
from tqdm.auto import tqdm

CLASSES = ['car_horn','children_playing','dog_bark','engine_idling','gun_shot','siren','drilling']
VI = ['Còi xe','Trẻ em vui chơi','Chó sủa','Động cơ chạy không tải','Tiếng súng','Còi báo động','Tiếng khoan']
MODEL_NAMES = {'cnn4':'CNN 4 khối', 'rescnn':'CNN residual tự xây dựng',
               'ast':'AST pretrained fine-tune', 'ensemble':'Ensemble xác suất'}
_PREDICTION_CACHE_LOCK = threading.RLock()
_PREDICTION_MODELS = {}
_AST_PROCESSORS = {}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def dump_json(obj, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    tmp.replace(path)

def atomic_save(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)

def load_pt(path, device='cpu'):
    # Only load checkpoints created by this notebook, never arbitrary untrusted pickle files.
    return torch.load(path,map_location=device,weights_only=True)

def save_fig(fig,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(); fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    display(fig); plt.close(fig)

def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def device_check(cfg,root):
    import subprocess
    diag={'python':sys.version,'torch':str(torch.__version__),'torch_cuda':torch.version.cuda,
          'cuda_available':torch.cuda.is_available(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
    try: diag['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=20).stdout
    except Exception as e: diag['nvidia_smi']=str(e)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        try:
            t=torch.randn(64,64,device=device); _=(t@t).sum().item(); torch.cuda.synchronize()
            diag['gpu']=torch.cuda.get_device_name(0); del t
        except Exception as e:
            diag['cuda_test_error']=str(e); device=torch.device('cpu')
    dump_json(diag,root/'outputs/logs/environment.json'); display(pd.DataFrame([diag]).T)
    if device.type=='cpu' and cfg['require_gpu']:
        raise RuntimeError('CUDA chưa hoạt động. Chọn Settings > Accelerator > GPU, khởi động lại session sạch và chạy từ đầu. '
                           'Nếu nvidia-smi có GPU nhưng torch_cuda=None: môi trường đã cài PyTorch CPU; tạo session/notebook mới. '
                           'Notebook này không cài đè torch. Chỉ đặt require_gpu=False để kiểm thử CPU, không để huấn luyện AST đầy đủ.')
    print('Thiết bị sử dụng:',device)
    return device


def discover_data(input_root,explicit_csv,root):
    """Only metadata-backed UrbanSound8K audio; demo files cannot become training samples."""
    required={'slice_file_name','fold','class'}
    candidates=[Path(explicit_csv)] if explicit_csv else list(Path(input_root).rglob('UrbanSound8K.csv'))
    valid=[]
    for p in candidates:
        try:
            d=pd.read_csv(p)
            if required.issubset(d.columns): valid.append((p,d))
        except Exception: pass
    if not valid:
        input_dirs=sorted(str(p) for p in Path(input_root).iterdir()) if Path(input_root).is_dir() else []
        raise FileNotFoundError('Chưa thấy UrbanSound8K.csv đúng định dạng. Add Input bộ UrbanSound8K đầy đủ (audio + metadata), '
                                'hoặc đặt CFG[data_csv]. Thư mục drilling chỉ dùng demo, không phải bộ huấn luyện. '
                                f'Các thư mục Input hiện thấy: {input_dirs[:20]}')
    if len(valid)>1:
        display(pd.DataFrame({'metadata':[str(p) for p,d in valid],'rows':[len(d) for p,d in valid]}))
        raise ValueError('Có nhiều metadata UrbanSound8K. Điền CFG[data_csv] đúng bản muốn sử dụng để tránh chọn nhầm.')
    csv,df=valid[0]; df=df[df['class'].isin(CLASSES)].copy()
    df['label']=df['class'].map({c:i for i,c in enumerate(CLASSES)}).astype(int)
    df['fold']=pd.to_numeric(df['fold'],errors='raise').astype(int)
    if not df.fold.between(1,10).all(): raise ValueError('Cột fold phải thuộc 1..10.')
    def canon(v):
        s=str(v)
        return str(int(float(s))) if s.replace('.','',1).isdigit() else s
    # fsID denotes original Freesound source, not the class ID or fold ID.
    df['source_id']=(df['fsID'].map(canon) if 'fsID' in df else df.slice_file_name.str.split('-').str[0])
    df=df.drop_duplicates('slice_file_name').reset_index(drop=True)
    index={}
    for p in Path(input_root).rglob('*.wav'): index.setdefault(p.name,[]).append(p)
    paths=[]
    for r in df.itertuples():
        preferred=csv.parent.parent/'audio'/f'fold{r.fold}'/r.slice_file_name
        if preferred.is_file(): p=preferred
        else:
            hits=index.get(r.slice_file_name,[])
            foldhits=[p for p in hits if p.parent.name==f'fold{r.fold}']
            hits=foldhits or hits
            if not hits: raise FileNotFoundError(f'Thiếu WAV được metadata tham chiếu: {r.slice_file_name}')
            if len(hits)>1 and len({sha_file(p) for p in hits})>1:
                raise ValueError(f'Nhiều file khác nội dung trùng tên {r.slice_file_name}; chỉ gắn một bộ dữ liệu.')
            p=sorted(hits)[0]
        paths.append(str(p))
    df['path']=paths
    df['duration']=np.nan; df['sample_rate']=0; df['channels']=0; df['sha256']=''
    rejected=[]
    for i,r in tqdm(df.iterrows(),total=len(df),desc='Kiểm tra WAV + SHA256'):
        try:
            info=sf.info(r.path)
            if info.frames<=0: raise ValueError('Âm thanh rỗng')
            df.loc[i,['duration','sample_rate','channels','sha256']]=[info.duration,info.samplerate,info.channels,sha_file(r.path)]
        except Exception as e: rejected.append({'index':i,'file':r.slice_file_name,'error':str(e)})
    pd.DataFrame(rejected,columns=['index','file','error']).to_csv(root/'outputs/data/rejected_audio.csv',index=False)
    if rejected: df=df.drop([r['index'] for r in rejected]).reset_index(drop=True)
    missing=set(CLASSES)-set(df['class'])
    if missing: raise ValueError(f'Thiếu dữ liệu lớp: {missing}')
    # Connected components link both source IDs and exact duplicate file bytes.
    parent=list(range(len(df)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for col in ['source_id','sha256']:
        first={}
        for i,key in enumerate(df[col]):
            if key in first: union(i,first[key])
            else: first[key]=i
    df['group']=[find(i) for i in range(len(df))]
    spans=df.groupby('source_id').fold.nunique(); cross=set(spans[spans>1].index)
    df[df.source_id.isin(cross)].to_csv(root/'outputs/data/source_cross_fold.csv',index=False)
    df[df.duplicated('sha256',keep=False)].to_csv(root/'outputs/data/exact_duplicates.csv',index=False)
    df.to_csv(root/'outputs/data/metadata_valid.csv',index=False)
    audit={'metadata':str(csv),'rows':len(df),'rejected':len(rejected),'sources_spanning_folds':len(cross),
           'exact_duplicate_rows':int(df.duplicated('sha256',keep=False).sum()),'excluded_classes':['air_conditioner','jackhammer','street_music']}
    dump_json(audit,root/'outputs/data/audit.json'); display(pd.DataFrame([audit]).T)
    if cross: print('Có nguồn xuất hiện ở nhiều fold. Sẽ purge theo thứ tự test > validation > train; không dừng bằng AssertionError.')
    return df,audit

def split_fold(df,test_fold,cfg,folder):
    val_fold=test_fold%10+1
    test=df[df.fold==test_fold].copy()
    val0=df[df.fold==val_fold].copy(); train0=df[~df.fold.isin([test_fold,val_fold])].copy()
    val=val0[~val0.group.isin(set(test.group))].copy()
    train=train0[~train0.group.isin(set(test.group)|set(val.group))].copy()
    parts={'train':train,'val':val,'test':test}
    if cfg['mode']=='smoke':
        parts={k:v.groupby('label',group_keys=False).head(cfg['smoke_per_class']).copy() for k,v in parts.items()}
    folder.mkdir(parents=True,exist_ok=True)
    purged=pd.concat([val0[~val0.index.isin(val.index)].assign(reason='source/duplicate overlap with test'),
                     train0[~train0.index.isin(train.index)].assign(reason='source/duplicate overlap with val or test')])
    purged.to_csv(folder/'purged.csv',index=False)
    for k,v in parts.items():
        v=v.sort_values('slice_file_name').reset_index(drop=True); parts[k]=v
        v.to_csv(folder/f'{k}.csv',index=False)
        if set(v.label)!=set(range(7)):
            raise ValueError(f'{k} sau purge thiếu lớp. Xem {folder}/purged.csv và chọn fold test khác; không tự trộn nguồn.')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if set(parts[a].group)&set(parts[b].group): raise RuntimeError(f'Rò rỉ nhóm {a}/{b}')
    dump_json({'test_fold':test_fold,'val_fold':val_fold,'purged_rows':len(purged),'counts':{k:len(v) for k,v in parts.items()},
               'protocol':'official fold holdout + source/duplicate purging; not original benchmark if purged', 'mode':cfg['mode']},folder/'split.json')
    return parts

def load_audio(path,sr=16000,seconds=4.0):
    y,orig=sf.read(str(path),always_2d=True,dtype='float32')
    if not np.isfinite(y).all(): raise ValueError(f'NaN/Inf trong âm thanh: {path}')
    y=y.mean(axis=1)
    if orig!=sr: y=librosa.resample(y,orig_sr=orig,target_sr=sr)
    if seconds is not None:
        n=int(sr*seconds)
        if len(y)>n: y=y[(len(y)-n)//2:(len(y)-n)//2+n]
        else: y=np.pad(y,(0,n-len(y)))
    return np.asarray(y,dtype=np.float32)

def logmel(y,cfg):
    m=librosa.feature.melspectrogram(y=y,sr=cfg['sr'],n_fft=1024,hop_length=256,
                                    n_mels=64,fmin=20,fmax=cfg['sr']/2,power=2)
    return librosa.power_to_db(m,ref=1.0,top_db=80).astype(np.float32)

def prepare_features(df,cfg,root,processor=None):
    cache=root/'cache'; cache.mkdir(exist_ok=True)
    sig=hashlib.sha256((df[['slice_file_name','sha256']].to_json()+json.dumps({'sr':cfg['sr'],'seconds':cfg['seconds'],
               'version':'logmel-v1','ast':cfg['ast_id'] if processor else None})).encode()).hexdigest()[:16]
    folder=cache/sig; folder.mkdir(exist_ok=True)
    melpaths=[]; astpaths=[]
    for r in tqdm(df.itertuples(),total=len(df),desc='Đặc trưng cố định (augmentation chỉ train)'):
        mpath=folder/(r.slice_file_name+'.mel.npy'); apath=folder/(r.slice_file_name+'.ast.npy')
        if not mpath.exists() or (processor is not None and not apath.exists()):
            y=load_audio(r.path,cfg['sr'],cfg['seconds'])
            if not mpath.exists(): np.save(mpath,logmel(y,cfg))
            if processor is not None and not apath.exists():
                # Force numpy backend in bootstrap if incompatible torchaudio is installed.
                a=processor(y,sampling_rate=cfg['sr'],return_tensors='np')['input_values'][0]
                if a.shape!=(1024,128): raise ValueError(f'AST shape bất thường: {a.shape}')
                np.save(apath,a.astype(np.float32))
        melpaths.append(str(mpath)); astpaths.append(str(apath) if processor else '')
    df=df.copy(); df['mel_path']=melpaths; df['ast_path']=astpaths
    df.to_csv(root/'outputs/data/metadata_features.csv',index=False)
    return df

def fit_stats(train):
    total=sq=0.; n=0
    for p in train.mel_path:
        x=np.load(p).astype(np.float64); total+=x.sum(); sq+=(x*x).sum(); n+=x.size
    mean=total/n; std=max(math.sqrt(max(sq/n-mean*mean,0)),1e-6)
    return {'mean':float(mean),'std':float(std),'fitted_on':'train only'}

class AudioDataset(Dataset):
    def __init__(self,df,kind,stats,augment=False):
        self.df=df.reset_index(drop=True); self.kind=kind; self.stats=stats; self.augment=augment
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        if self.kind=='ast': x=torch.from_numpy(np.load(r.ast_path).copy()) # time x mel
        else:
            x=torch.from_numpy(np.load(r.mel_path).copy())
            x=(x-self.stats['mean'])/self.stats['std']; x=x.unsqueeze(0)
        if self.augment:
            td,fd=(0,1) if self.kind=='ast' else (2,1)
            x=torch.roll(x,random.randint(-12,12),dims=td)
            for dim in [td,fd]:
                width=random.randint(0,max(1,int(x.shape[dim]*.08)))
                start=random.randint(0,x.shape[dim]-width)
                sl=[slice(None)]*x.ndim; sl[dim]=slice(start,start+width); x[tuple(sl)]=0
        return x,int(r.label)

class CNN4(nn.Module):
    """Adaptation of Walden: 4 blocks 32/64/128/256; mono 16k, not exact paper reproduction."""
    def __init__(self,n_classes=7):
        super().__init__(); blocks=[]; cin=1
        for i,(cout,k) in enumerate(zip([32,64,128,256],[(3,5),(3,5),(5,5),(5,5)])):
            blocks.extend([nn.Conv2d(cin,cout,k,stride=2 if i==0 else 1,padding=2),nn.ReLU(),nn.BatchNorm2d(cout)])
            cin=cout
        self.features=nn.Sequential(*blocks); self.pool=nn.AdaptiveAvgPool2d(1)
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.3),nn.Linear(256,n_classes))
    def forward(self,x): return self.head(self.pool(self.features(x)))

class ResidualBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.main=nn.Sequential(nn.Conv2d(cin,cout,3,stride,1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(),
                                nn.Conv2d(cout,cout,3,1,1,bias=False),nn.BatchNorm2d(cout))
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout))
    def forward(self,x): return torch.relu(self.main(x)+self.skip(x))

class ResCNN(nn.Module):
    def __init__(self,n_classes=7):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,32,5,2,2),nn.BatchNorm2d(32),nn.ReLU(),
            ResidualBlock(32,32),ResidualBlock(32,64,2),ResidualBlock(64,128,2),ResidualBlock(128,256,2),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.35),nn.Linear(256,n_classes))
    def forward(self,x): return self.net(x)

class CheckpointCNN4(nn.Module):
    def __init__(self,n_classes=7):
        super().__init__()
        channels=(32,64,128,256)
        blocks=[]
        in_channels=1
        for out_channels in channels:
            blocks.append(nn.Sequential(
                nn.Conv2d(in_channels,out_channels,3,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            in_channels=out_channels
        self.features=nn.Sequential(*blocks)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Dropout(.3),
            nn.Linear(128,n_classes),
        )

    def forward(self,x):
        return self.head(self.pool(self.features(x)))

class CheckpointResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,reduction=8,stride=2):
        super().__init__()
        self.c1=nn.Conv2d(in_channels,out_channels,3,stride=stride,padding=1,bias=False)
        self.b1=nn.BatchNorm2d(out_channels)
        self.c2=nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False)
        self.b2=nn.BatchNorm2d(out_channels)
        self.se=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels,max(out_channels//reduction,1),1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_channels//reduction,1),out_channels,1),
            nn.Sigmoid(),
        )
        self.short=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,stride=stride,bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self,x):
        residual=self.short(x)
        x=torch.relu(self.b1(self.c1(x)))
        x=self.b2(self.c2(x))
        return torch.relu(x*self.se(x)+residual)

class CheckpointResCNN(nn.Module):
    def __init__(self,n_classes=7):
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(1,32,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.s1=CheckpointResidualBlock(32,64,reduction=8,stride=2)
        self.s2=CheckpointResidualBlock(64,128,reduction=8,stride=2)
        self.s3=CheckpointResidualBlock(128,256,reduction=8,stride=2)
        self.s4=CheckpointResidualBlock(256,256,reduction=8,stride=1)
        self.head=nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(.3),
            nn.Linear(512,128),
            nn.ReLU(inplace=True),
            nn.Dropout(.3),
            nn.Linear(128,n_classes),
        )

    def forward(self,x):
        x=self.stem(x)
        x=self.s4(self.s3(self.s2(self.s1(x))))
        pooled=torch.cat(
            (nn.functional.adaptive_avg_pool2d(x,1),nn.functional.adaptive_max_pool2d(x,1)),
            dim=1,
        )
        return self.head(pooled.flatten(1))

def import_ast():
    # AST only needs audio numpy preprocessing; prevent loading unrelated broken torchvision/torchaudio binaries.
    import transformers.utils.import_utils as iu
    import transformers.utils as tu
    iu._torchvision_available=False; iu._torchaudio_available=False
    # Transformers 5 checks torchaudio through this public helper before importing its optional binary backend.
    tu.is_speech_available=lambda: False
    tu.is_torchaudio_available=lambda: False
    tu.is_torchvision_available=lambda: False
    from transformers import ASTFeatureExtractor, ASTForAudioClassification, ASTConfig
    return ASTFeatureExtractor,ASTForAudioClassification,ASTConfig

def prepare_ast(cfg,root):
    Feature,Model,Config=import_ast()
    source=cfg['ast_local_path'] or cfg['ast_id']
    processor=Feature.from_pretrained(source)
    processor.save_pretrained(root/'configs/ast_processor')
    # Verify pretrained weights BEFORE the long CNN training stage. No random-weight fallback.
    model=Model.from_pretrained(source,use_safetensors=True,attn_implementation='eager')
    model.config.save_pretrained(root/'configs/ast_original')
    dump_json({'source':source,'revision':getattr(model.config,'_commit_hash',None),'pretrained_verified':True},root/'configs/pretrained_provenance.json')
    del model; gc.collect()
    return processor

def build_model(kind,cfg,device,root=None,from_checkpoint=False):
    flat_checkpoint=(
        from_checkpoint and root is not None
        and prediction_checkpoint_path(root,cfg.get('test_folds',[10])[0],kind).parent
        == Path(root)/'outputs/checkpoints'
    )
    if kind=='cnn4':
        model=CheckpointCNN4() if flat_checkpoint else CNN4()
    elif kind=='rescnn':
        model=CheckpointResCNN() if flat_checkpoint else ResCNN()
    else:
        Feature,Model,Config=import_ast()
        if from_checkpoint:
            config=Config.from_pretrained(Path(root)/'configs/ast_7class')
            model=Model(config)
        else:
            model=Model.from_pretrained(cfg['ast_local_path'] or cfg['ast_id'],use_safetensors=True,
                      num_labels=7,id2label=dict(enumerate(VI)),label2id={s:i for i,s in enumerate(VI)},
                      ignore_mismatched_sizes=True,attn_implementation='eager')
            model.config.save_pretrained(Path(root)/'configs/ast_7class')
        if cfg.get('gradient_checkpointing',True): model.gradient_checkpointing_enable()
    return model.to(device)

def prediction_checkpoint_path(root,fold,kind):
    root=Path(root)
    nested=root/'outputs/checkpoints'/f'fold{fold:02d}'/kind/'best.pt'
    if nested.is_file():
        return nested
    return root/'outputs/checkpoints'/f'{kind}_best.pt'

def available_checkpoint_kinds(root,fold,cfg=None):
    root=Path(root)
    if cfg is None:
        cfg=json.loads((root/'configs/config.json').read_text(encoding='utf-8'))
    kinds=[]
    for kind in cfg['models']:
        checkpoint=prediction_checkpoint_path(root,fold,kind)
        if not checkpoint.is_file():
            continue
        kinds.append(kind)
    return kinds

def _get_prediction_model(kind,cfg,root,fold,device):
    checkpoint=prediction_checkpoint_path(root,fold,kind)
    stat=checkpoint.stat()
    key=(str(checkpoint.resolve()),stat.st_mtime_ns,stat.st_size,str(device))
    with _PREDICTION_CACHE_LOCK:
        model=_PREDICTION_MODELS.get(key)
        if model is None:
            model=build_model(kind,cfg,device,root,from_checkpoint=True)
            state=load_pt(checkpoint)
            weights=state['model']
            if weights and all(name.startswith('model.') for name in weights):
                weights={name[6:]:value for name,value in weights.items()}
            model.load_state_dict(weights)
            del state
            if kind=='ast' and hasattr(model,'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
            model.eval()
            _PREDICTION_MODELS[key]=model
        return model

def _get_ast_processor(root):
    key=str((root/'configs/ast_processor').resolve())
    with _PREDICTION_CACHE_LOCK:
        processor=_AST_PROCESSORS.get(key)
        if processor is None:
            Feature,_,_=import_ast()
            processor=Feature.from_pretrained(root/'configs/ast_processor',local_files_only=True)
            _AST_PROCESSORS[key]=processor
        return processor

def warmup_prediction_model(kind,cfg,root,fold,device):
    model=_get_prediction_model(kind,cfg,root,fold,device)
    logmel(np.zeros(int(cfg['seconds']*cfg['sr']),dtype=np.float32),cfg)
    if kind=='ast':
        sample=torch.zeros((1,1024,128),device=device)
    else:
        frames=int(cfg['seconds']*cfg['sr']/256)+1
        sample=torch.zeros((1,1,64,frames),device=device)
    with torch.inference_mode():
        with amp_context(device,cfg['amp']):
            logits_of(model,sample,kind)
    if device.type=='cuda':
        torch.cuda.synchronize(device)

def logits_of(model,x,kind):
    return model(input_values=x).logits if kind=='ast' else model(x)

def audio_windows(y, window_samples, hop_samples):
    if len(y)==0: raise ValueError('File demo rỗng')
    starts=list(range(0,max(1,len(y)-window_samples+1),hop_samples))
    if len(y)>window_samples and starts[-1]!=len(y)-window_samples:
        starts.append(len(y)-window_samples)
    return [np.pad(y[s:s+window_samples],(0,max(0,window_samples-len(y[s:s+window_samples]))))
            for s in starts]

def make_loader(df,kind,stats,batch,train=False,seed=42):
    gen=torch.Generator().manual_seed(seed)
    return DataLoader(AudioDataset(df,kind,stats,train),batch_size=batch,shuffle=train,
                      num_workers=0,pin_memory=torch.cuda.is_available(),generator=gen,drop_last=False)

def amp_context(device,enabled):
    return torch.autocast(device_type='cuda',dtype=torch.float16) if device.type=='cuda' and enabled else nullcontext()

def scores(y,p):
    pred=p.argmax(1); pr,re,f1,_=precision_recall_fscore_support(y,pred,labels=list(range(7)),average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_precision':float(pr),'macro_recall':float(re),'macro_f1':float(f1)}

@torch.no_grad()
def evaluate(model,loader,kind,device,amp=True):
    model.eval(); ys=[]; probs=[]; total=0.; n=0
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        with amp_context(device,amp): out=logits_of(model,x,kind); loss=nn.functional.cross_entropy(out,y)
        total+=loss.item()*len(y); n+=len(y); ys.extend(y.cpu().tolist()); probs.append(out.float().softmax(1).cpu().numpy())
    p=np.concatenate(probs); y=np.asarray(ys); m=scores(y,p); m['loss']=total/max(n,1)
    return m,y,p

def train_one(kind,parts,stats,cfg,root,fold,device):
    folder=root/'outputs/checkpoints'/f'fold{fold:02d}'/kind; folder.mkdir(parents=True,exist_ok=True)
    # Any change to data, labels, or relevant config invalidates reuse.
    ignored={'resume_from','resume','make_zip','demo_path','input_root','data_csv','ast_local_path','require_gpu','demo_threshold','silence_rms'}
    train_cfg={k:v for k,v in cfg.items() if k not in ignored}
    sig=hashlib.sha256(('engine-v1'+json.dumps(train_cfg,sort_keys=True)+''.join(v[['slice_file_name','sha256']].to_json() for v in parts.values())+kind).encode()).hexdigest()
    status=folder/'complete.json'
    if status.exists() and cfg['resume']:
        st=json.loads(status.read_text())
        if st['signature']!=sig: raise ValueError(f'Output cũ không cùng cấu hình: {folder}. Đổi experiment_name.')
        return st
    seed_all(cfg['seed']+fold)
    model=build_model(kind,cfg,device,root)
    isast=kind=='ast'; batch=cfg['ast_batch'] if isast else cfg['cnn_batch']
    accum=cfg['ast_accum'] if isast else 1
    epochs=1 if cfg['mode']=='smoke' else (cfg['ast_epochs'] if isast else cfg['cnn_epochs'])
    lr=cfg['ast_lr'] if isast else cfg['cnn_lr']
    if isast:
        head=list(model.classifier.parameters()); headids={id(p) for p in head}
        groups=[{'params':[p for p in model.parameters() if id(p) not in headids],'lr':lr}, {'params':head,'lr':lr*10}]
    else: groups=model.parameters()
    opt=torch.optim.AdamW(groups,lr=lr,weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(epochs,1))
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and cfg['amp']))
    criterion=nn.CrossEntropyLoss(label_smoothing=.05)
    val_loader=make_loader(parts['val'],kind,stats,batch)
    history=[]; best=-1.; bad=0; start=0; elapsed=0.
    last=folder/'last.pt'
    if last.exists() and cfg['resume']:
        ck=load_pt(last)
        if ck['signature']!=sig: raise ValueError('Checkpoint khác cấu hình/dữ liệu. Đổi experiment_name.')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler']); history=ck['history']; start=ck['epoch']+1; best=ck['best']; bad=ck['bad']; elapsed=ck['elapsed']
        del ck
    count=sum(p.numel() for p in model.parameters()); print(MODEL_NAMES[kind],f'{count:,} tham số; batch={batch}; accumulation={accum}')
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    for epoch in range(start,epochs):
        if bad>=cfg['patience']: break
        seed_all(cfg['seed']+fold*1000+epoch)
        loader=make_loader(parts['train'],kind,stats,batch,True,cfg['seed']+epoch)
        model.train(); opt.zero_grad(set_to_none=True); t=time.perf_counter(); total=0.; correct=0; n=0; skipped=0
        try:
            for step,(x,y) in enumerate(tqdm(loader,desc=f'{kind} fold {fold} epoch {epoch+1}/{epochs}',leave=False)):
                x=x.to(device); y=y.to(device)
                window_start=(step//accum)*accum; denom=min(accum,len(loader)-window_start)
                with amp_context(device,cfg['amp']):
                    out=logits_of(model,x,kind); rawloss=criterion(out,y); loss=rawloss/denom
                if not torch.isfinite(loss): raise FloatingPointError('Loss NaN/Inf: thử amp=False, giảm learning rate.')
                scaler.scale(loss).backward()
                total+=rawloss.item()*len(y); correct+=(out.argmax(1)==y).sum().item(); n+=len(y)
                if (step+1)%accum==0 or step+1==len(loader):
                    scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError as e:
            opt.zero_grad(set_to_none=True); del model; gc.collect(); torch.cuda.empty_cache()
            raise RuntimeError('GPU hết VRAM. Giảm ast_batch xuống 1 hoặc cnn_batch xuống 8, giữ ast_accum=16; '
                               'đổi experiment_name khi đổi cấu hình. AST đã bật gradient checkpointing.') from e
        val,_,_=evaluate(model,val_loader,kind,device,cfg['amp']); elapsed+=time.perf_counter()-t
        row={'epoch':epoch+1,'train_loss':total/n,'train_accuracy':correct/n,'val_loss':val['loss'],
             'val_accuracy':val['accuracy'],'val_macro_f1':val['macro_f1'],'learning_rate':opt.param_groups[0]['lr'],'seconds':elapsed}
        history.append(row); print(row)
        if val['macro_f1']>best+1e-8:
            best=val['macro_f1']; bad=0
            atomic_save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'kind':kind,'epoch':epoch+1,
                         'signature':sig,'stats':stats,'classes':CLASSES,'labels_vi':VI},folder/'best.pt')
        else: bad+=1
        scheduler.step()
        atomic_save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),
                     'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'bad':bad,
                     'history':history,'signature':sig,'elapsed':elapsed},last)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    ck=load_pt(folder/'best.pt'); model.load_state_dict(ck['model']); del ck
    val,y,p=evaluate(model,val_loader,kind,device,cfg['amp'])
    np.savez_compressed(folder/'val_predictions.npz',y=y,p=p)
    st={'signature':sig,'kind':kind,'fold':fold,'parameters':count,'training_seconds':elapsed,
        'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0,
        'checkpoint_mb':(folder/'best.pt').stat().st_size/2**20,'val':val,'mode':cfg['mode']}
    dump_json(st,status); del model,opt,scheduler,scaler; gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    return st

def training_plots(kind,fold,root):
    folder=root/'outputs/checkpoints'/f'fold{fold:02d}'/kind
    hist=pd.read_csv(folder/'history.csv')
    fig,ax=plt.subplots(1,3,figsize=(16,4))
    ax[0].plot(hist.epoch,hist.train_loss,label='Train'); ax[0].plot(hist.epoch,hist.val_loss,label='Validation'); ax[0].set_title('Loss')
    ax[1].plot(hist.epoch,hist.train_accuracy,label='Train'); ax[1].plot(hist.epoch,hist.val_accuracy,label='Validation'); ax[1].set_title('Accuracy')
    ax[2].plot(hist.epoch,hist.val_macro_f1,label='Validation'); ax[2].set_title('Macro-F1')
    for a in ax: a.set_xlabel('Epoch'); a.legend(); a.grid(alpha=.2)
    fig.suptitle(f'{MODEL_NAMES[kind]} | Fold {fold}')
    save_fig(fig,root/f'outputs/figures/fold{fold:02d}/{kind}_history')

def choose_ensemble(kinds,fold,root):
    import itertools
    base=root/'outputs/checkpoints'/f'fold{fold:02d}'
    arrays=[np.load(base/k/'val_predictions.npz') for k in kinds]
    y=arrays[0]['y']; ps=[a['p'] for a in arrays]
    if not all(np.array_equal(a['y'],y) for a in arrays): raise ValueError('Thứ tự validation không đồng nhất')
    rows=[]
    for units in itertools.product(range(5),repeat=len(kinds)):
        if sum(units)!=4: continue
        w=np.array(units)/4.; p=sum(a*b for a,b in zip(w,ps)); m=scores(y,p)
        rows.append({'weights':w.tolist(),**m})
    table=pd.DataFrame(rows).sort_values(['macro_f1','accuracy'],ascending=False,kind='stable').reset_index(drop=True)
    best=table.iloc[0]; spec={'kinds':kinds,'weights':best.weights,'val_macro_f1':float(best.macro_f1),'selected_on':'validation only'}
    out=root/f'outputs/metrics/fold{fold:02d}'; out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/'ensemble_search_validation.csv',index=False); dump_json(spec,out/'ensemble.json')
    display(table.head(10)); return spec


def report_predictions(y,p,df,kind,fold,root):
    out=root/f'outputs/metrics/fold{fold:02d}'; out.mkdir(parents=True,exist_ok=True)
    predout=root/f'outputs/predictions/fold{fold:02d}'; predout.mkdir(parents=True,exist_ok=True)
    pred=p.argmax(1); m=scores(y,p)
    rep=pd.DataFrame(classification_report(y,pred,labels=list(range(7)),target_names=VI,output_dict=True,zero_division=0)).T
    rep.to_csv(out/f'{kind}_classification_report.csv',encoding='utf-8-sig'); display(rep.round(4))
    result=df[['slice_file_name','source_id','fold','class','path']].copy()
    result['true_label_vi']=[VI[i] for i in y]; result['predicted_label_vi']=[VI[i] for i in pred]
    result['confidence']=p.max(1); result['correct']=pred==y
    for i,c in enumerate(CLASSES): result['p_'+c]=p[:,i]
    result.to_csv(predout/f'{kind}_test_predictions.csv',index=False,encoding='utf-8-sig')
    result[~result.correct].sort_values('confidence',ascending=False).to_csv(predout/f'{kind}_errors.csv',index=False,encoding='utf-8-sig')
    cm=confusion_matrix(y,pred,labels=list(range(7))); norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
    pd.DataFrame(cm,index=VI,columns=VI).to_csv(out/f'{kind}_confusion_counts.csv',encoding='utf-8-sig')
    fig,ax=plt.subplots(1,2,figsize=(18,6))
    for a,data,fmt,title in [(ax[0],cm,'d','Số lượng'),(ax[1],norm,'.2f','Chuẩn hóa theo nhãn thật')]:
        sns.heatmap(data,annot=True,fmt=fmt,xticklabels=VI,yticklabels=VI,cmap='Blues',ax=a)
        a.set(title=title,xlabel='Dự đoán',ylabel='Nhãn thật'); a.tick_params(axis='x',rotation=45)
    fig.suptitle(f'{MODEL_NAMES[kind]} — test fold {fold}')
    save_fig(fig,root/f'outputs/figures/fold{fold:02d}/{kind}_confusion')
    fig,ax=plt.subplots(1,2,figsize=(15,5)); curves={}; curve_scores=[]
    for i,label in enumerate(VI):
        binary=(y==i).astype(int)
        if binary.min()==binary.max(): continue
        fpr,tpr,_=roc_curve(binary,p[:,i]); pr,re,_=precision_recall_curve(binary,p[:,i])
        area=auc(fpr,tpr); ap=average_precision_score(binary,p[:,i])
        ax[0].plot(fpr,tpr,label=f'{label}: {area:.3f}'); ax[1].plot(re,pr,label=f'{label}: {ap:.3f}')
        curves[f'{i}_fpr']=fpr; curves[f'{i}_tpr']=tpr; curves[f'{i}_precision']=pr; curves[f'{i}_recall']=re
        curve_scores.append({'class':CLASSES[i],'roc_auc_ovr':area,'average_precision':ap})
    ax[0].plot([0,1],[0,1],'k--'); ax[0].set(title='ROC one-vs-rest',xlabel='FPR',ylabel='TPR')
    ax[1].set(title='Precision–Recall',xlabel='Recall',ylabel='Precision')
    for a in ax: a.legend(fontsize=8); a.grid(alpha=.2)
    save_fig(fig,root/f'outputs/figures/fold{fold:02d}/{kind}_roc_pr')
    np.savez_compressed(out/f'{kind}_curves.npz',**curves)
    pd.DataFrame(curve_scores).to_csv(out/f'{kind}_roc_ap.csv',index=False)
    dump_json(m,out/f'{kind}_test_metrics.json')
    display(Markdown('**Các dự đoán sai có độ tự tin cao:**')); display(result[~result.correct].sort_values('confidence',ascending=False).head(8))
    return m


def evaluate_fold(parts,stats,cfg,root,fold,device,kinds,ensemble,states):
    valrows=[{'model':k,**states[k]['val']} for k in kinds]
    valrows.append({'model':'ensemble','macro_f1':ensemble['val_macro_f1']})
    valtable=pd.DataFrame(valrows).sort_values('macro_f1',ascending=False,kind='stable')
    chosen=valtable.iloc[0]['model']
    out=root/f'outputs/metrics/fold{fold:02d}'; out.mkdir(parents=True,exist_ok=True)
    valtable.to_csv(out/'validation_comparison.csv',index=False)
    dump_json({'selected_model':chosen,'criterion':'validation macro-F1','test_not_used_for_selection':True},out/'selection.json')
    display(Markdown(f'**Mô hình chọn trước khi đánh giá test: {MODEL_NAMES[chosen]}**')); display(valtable)
    predictions={}; rows=[]
    for kind in kinds:
        folder=root/f'outputs/checkpoints/fold{fold:02d}'/kind
        model=build_model(kind,cfg,device,root,from_checkpoint=True)
        model.load_state_dict(load_pt(folder/'best.pt')['model'])
        batch=cfg['ast_batch'] if kind=='ast' else cfg['cnn_batch']
        loader=make_loader(parts['test'],kind,stats,batch)
        if device.type=='cuda': torch.cuda.synchronize()
        start=time.perf_counter(); metrics,y,p=evaluate(model,loader,kind,device,cfg['amp'])
        if device.type=='cuda': torch.cuda.synchronize()
        ms=1000*(time.perf_counter()-start)/len(y)
        predictions[kind]=p
        m=report_predictions(y,p,parts['test'],kind,fold,root)
        rows.append({'fold':fold,'model':kind,**m,'val_macro_f1':states[kind]['val']['macro_f1'],
                     'training_seconds':states[kind]['training_seconds'],'parameters':states[kind]['parameters'],
                     'checkpoint_mb':states[kind]['checkpoint_mb'],'peak_vram_mb':states[kind]['peak_vram_mb'],
                     'eval_ms_per_clip_with_loading':ms,'selected_by_validation':kind==chosen})
        del model; gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    ep=sum(w*predictions[k] for k,w in zip(kinds,ensemble['weights']))
    m=report_predictions(y,ep,parts['test'],'ensemble',fold,root)
    active={k for k,w in zip(kinds,ensemble['weights']) if w>0}
    rows.append({'fold':fold,'model':'ensemble',**m,'val_macro_f1':ensemble['val_macro_f1'],
                 'parameters':sum(states[k]['parameters'] for k in active),
                 'checkpoint_mb':sum(states[k]['checkpoint_mb'] for k in active),
                 'training_seconds':sum(states[k]['training_seconds'] for k in active),
                 'eval_ms_per_clip_with_loading':sum(r['eval_ms_per_clip_with_loading'] for r in rows if r['model'] in active),
                 'selected_by_validation':chosen=='ensemble'})
    df=pd.DataFrame(rows); df.to_csv(out/'test_comparison.csv',index=False)
    display(df.round(4))
    fig,ax=plt.subplots(figsize=(10,4)); df.set_index('model')[['accuracy','macro_f1']].plot.bar(ax=ax,ylim=(0,1))
    ax.set(title=f'Test fold {fold} — không dùng để chọn mô hình',ylabel='Điểm',xlabel='Mô hình')
    save_fig(fig,root/f'outputs/figures/fold{fold:02d}/comparison')
    return df


def run_experiments(df,cfg,root,device):
    allrows=[]
    kinds=cfg['models']
    for fold in cfg['test_folds']:
        parts=split_fold(df,fold,cfg,root/f'outputs/data/fold{fold:02d}')
        stats=fit_stats(parts['train']); dump_json(stats,root/f'configs/stats_fold{fold:02d}.json')
        counttable=pd.DataFrame({k:v['class'].value_counts().reindex(CLASSES,fill_value=0) for k,v in parts.items()})
        display(counttable); counttable.to_csv(root/f'outputs/data/fold{fold:02d}/counts.csv')
        fig,ax=plt.subplots(figsize=(10,4)); counttable.set_axis(VI).plot.bar(ax=ax)
        ax.set(title=f'Phân bố Train / Validation / Test — fold {fold}',ylabel='Số mẫu')
        save_fig(fig,root/f'outputs/figures/fold{fold:02d}/split_counts')
        states={}
        for kind in kinds:
            states[kind]=train_one(kind,parts,stats,cfg,root,fold,device)
            training_plots(kind,fold,root)
        ensemble=choose_ensemble(kinds,fold,root)
        allrows.append(evaluate_fold(parts,stats,cfg,root,fold,device,kinds,ensemble,states))
    results=pd.concat(allrows,ignore_index=True)
    results.to_csv(root/'outputs/metrics/all_fold_results.csv',index=False)
    summary=results.groupby('model')[['accuracy','macro_f1','macro_precision','macro_recall']].agg(['mean','std'])
    summary.to_csv(root/'outputs/metrics/mean_std.csv'); display(summary)
    if len(cfg['test_folds'])<10: print('Đây là kết quả holdout/partial folds, KHÔNG phải kết quả 10-fold CV đầy đủ.')
    return results


def eda(df,cfg,root):
    counts=df['class'].value_counts().reindex(CLASSES)
    table=pd.DataFrame({'Nhãn gốc':CLASSES,'Tiếng Việt':VI,'Số mẫu':counts.values})
    display(table); table.to_csv(root/'outputs/data/class_counts.csv',index=False,encoding='utf-8-sig')
    display(df[['duration','sample_rate','channels']].describe())
    df[['duration','sample_rate','channels']].describe().to_csv(root/'outputs/data/audio_statistics.csv')
    fig,ax=plt.subplots(1,2,figsize=(14,4))
    ax[0].bar(VI,counts.values,color=sns.color_palette('viridis',7)); ax[0].tick_params(axis='x',rotation=45); ax[0].set_title('Số lượng theo lớp')
    ax[1].hist(df.duration,bins=35,color='#168b96'); ax[1].set(title='Thời lượng gốc',xlabel='Giây',ylabel='Số file')
    save_fig(fig,root/'outputs/figures/eda_distribution')
    fig,axes=plt.subplots(7,2,figsize=(15,19))
    for i,c in enumerate(CLASSES):
        r=df[df['class']==c].iloc[0]; y=load_audio(r.path,cfg['sr'],cfg['seconds']); m=logmel(y,cfg)
        axes[i,0].plot(np.arange(len(y))/cfg['sr'],y,lw=.6); axes[i,0].set(title=VI[i],xlabel='Thời gian (s)',ylabel='Biên độ')
        axes[i,1].imshow(m,origin='lower',aspect='auto',cmap='magma'); axes[i,1].set(xlabel='Khung thời gian',ylabel='Mel bin')
        display(Markdown(f'**{VI[i]} — {r.slice_file_name}**')); display(Audio(y,rate=cfg['sr']))
    save_fig(fig,root/'outputs/figures/eda_waveform_logmel')


def demo_paths(path):
    p=Path(path)
    if p.is_file(): return [p]
    if p.is_dir(): return sorted(q for q in p.rglob('*') if q.suffix.lower() in ['.wav','.flac','.ogg','.mp3'])
    return []

@torch.inference_mode()
def predict_file(path,root,fold=None,device=None,model_choice=None,waveform=None):
    root=Path(root); cfg=json.loads((root/'configs/config.json').read_text())
    fold=fold or cfg['test_folds'][0]; device=device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    stats=json.loads((root/f'configs/stats_fold{fold:02d}.json').read_text())
    metrics=root/f'outputs/metrics/fold{fold:02d}'
    selection_path=metrics/'selection.json'
    selected=(json.loads(selection_path.read_text(encoding='utf-8'))['selected_model']
              if selection_path.is_file() else None)
    ensemble_path=metrics/'ensemble.json'
    ens=json.loads(ensemble_path.read_text(encoding='utf-8')) if ensemble_path.is_file() else None
    available=available_checkpoint_kinds(root,fold,cfg)
    chosen=model_choice or (selected if selected in available else None) or (available[0] if available else None)
    if chosen is None:
        raise FileNotFoundError(f'Không tìm thấy checkpoint tương thích trong {root/"outputs/checkpoints"}.')
    allowed=set(cfg['models'])|{'ensemble'}
    if chosen not in allowed: raise ValueError(f'Mô hình không hợp lệ: {chosen}')
    if chosen=='ensemble':
        kinds=(
            [k for k,w in zip(ens['kinds'],ens['weights']) if w>0]
            if ens is not None else available
        )
        missing=[kind for kind in kinds if kind not in available]
        if missing:
            raise FileNotFoundError(
                f'Ensemble cần checkpoint không có trong outputs/checkpoints: {missing}.'
            )
    else:
        kinds=[chosen]
    if not kinds:
        raise FileNotFoundError('Không có checkpoint nào để chạy ensemble.')
    # Window full audio; same 4-second representation and preprocessing as training.
    y=load_audio(path,cfg['sr'],None) if waveform is None else np.asarray(waveform,dtype=np.float32)
    n=int(cfg['seconds']*cfg['sr']); hop=n//2
    waves=audio_windows(y,n,hop)
    mel=torch.from_numpy(np.stack([logmel(w,cfg) for w in waves]))[:,None]
    mel=(mel-stats['mean'])/stats['std']
    ast_x=None; pred={}
    if 'ast' in kinds:
        proc=_get_ast_processor(root)
        ast_x=torch.tensor(proc(waves,sampling_rate=cfg['sr'],return_tensors='np')['input_values'])
    for kind in kinds:
        model=_get_prediction_model(kind,cfg,root,fold,device)
        x=ast_x if kind=='ast' else mel
        batch_size=1 if kind=='ast' else 16
        pp=[]
        for start in range(0,len(x),batch_size):
            batch=x[start:start+batch_size].to(device)
            with amp_context(device,cfg['amp']):
                out=logits_of(model,batch,kind)
            pp.append(out.float().softmax(1).cpu().numpy())
        pred[kind]=np.concatenate(pp,axis=0).mean(axis=0)
    if chosen=='ensemble':
        if ens is None:
            pred['ensemble']=sum(pred[k] for k in kinds)/len(kinds)
        else:
            pred['ensemble']=sum(w*pred[k] for k,w in zip(ens['kinds'],ens['weights']) if w>0)
    p=pred[chosen]; ix=int(p.argmax()); rms=float(np.sqrt(np.mean(y*y)))
    label=VI[ix] if p[ix]>=cfg['demo_threshold'] and rms>=cfg['silence_rms'] else 'Chưa xác định'
    return {'file':str(path),'selected_model':chosen,'label':label,'top1_class':CLASSES[ix],
            'confidence':float(p[ix]),'rms':rms,'windows':len(waves),
            'note':'Ngưỡng từ chối mang tính heuristic; chưa đánh giá khả năng nhận diện âm thanh ngoài tập lớp.',
            'probabilities':{k:{VI[i]:float(v) for i,v in enumerate(a)} for k,a in pred.items()}}


def show_demo(cfg,root,device):
    paths=demo_paths(cfg['demo_path'])
    if not paths:
        print('Không tìm thấy file demo tại',cfg['demo_path'],'. Các kết quả huấn luyện vẫn đã lưu. Add Input audio rồi chạy lại cell demo.'); return None
    print('Tìm thấy',len(paths),'file; đang demo file đầu tiên:',paths[0])
    if len(paths)>1: display(pd.DataFrame({'file':[str(p) for p in paths]}).head(30))
    result=predict_file(paths[0],root,device=device)
    y=load_audio(paths[0],cfg['sr'],None); display(Audio(y,rate=cfg['sr']))
    table=pd.DataFrame(result['probabilities']).T; display(table.round(4))
    display(Markdown(f"### Kết quả: {result['label']}\nMô hình: {result['selected_model']} · điểm top-1: {result['confidence']:.2%}"))
    fig,ax=plt.subplots(figsize=(10,4)); table.loc[result['selected_model']].plot.bar(ax=ax,color='#168b96',ylim=(0,1))
    ax.set(title='Điểm dự đoán demo — không phải độ chính xác',ylabel='Softmax trung bình cửa sổ')
    save_fig(fig,root/'outputs/figures/demo_probabilities')
    dump_json(result,root/'outputs/predictions/demo_result.json'); table.to_csv(root/'outputs/predictions/demo_probabilities.csv',encoding='utf-8-sig')
    return result


def finalize_report(cfg,root,results):
    import html, subprocess
    links=[]
    for p in sorted((root/'outputs/figures').rglob('*.png')):
        rel=p.relative_to(root/'outputs').as_posix()
        links.append(f'<h3>{html.escape(p.stem)}</h3><img loading="lazy" src="{rel}" style="max-width:100%">')
    protocol='10 fold đã chạy' if set(cfg['test_folds'])==set(range(1,11)) else f"Holdout/partial: test folds {cfg['test_folds']}"
    intro=f'''# Báo cáo phân loại âm thanh đô thị — 7 lớp

## Phạm vi
Mô hình: CNN4 tham khảo Walden, ResCNN tự xây dựng, AST pretrained fine-tune và ensemble.
Chỉ giữ car_horn, children_playing, dog_bark, engine_idling, gun_shot, siren, drilling.
Loại air_conditioner, jackhammer, street_music. Không gộp jackhammer vào drilling.

## Phương pháp và giới hạn so sánh
{protocol}. Chế độ: {cfg['mode']}. Train/validation/test cùng dữ liệu cho mọi mô hình; purge nguồn trùng ưu tiên test.
Nếu mode=smoke, mọi kết quả chỉ là kiểm thử kỹ thuật, không dùng để viết kết luận hiệu năng.
Chuẩn hóa CNN fit trên train; AST dùng chuẩn hóa pretrained AudioSet. Augmentation chỉ train.
CNN4 sử dụng 4 khối 32/64/128/256, kernels (3,5),(3,5),(5,5),(5,5), điều chỉnh mono 16 kHz và pooling toàn cục.
Bài Walden dùng stereo 44.1 kHz, tập 7 lớp có điều hòa thay cho drilling, chia 80/20 và báo cáo 97.82%.
Đây không phải tái lập chính xác bài báo. Không so trực tiếp accuracy giữa hai thiết lập.
Nguồn pretraining AST: AudioSet; chưa kiểm toán trùng nguồn giữa AudioSet và UrbanSound8K.
Các fold được purge không còn là benchmark UrbanSound8K nguyên bản. Xem split.json và purged.csv từng fold.
Chọn checkpoint, ensemble và mô hình demo bằng validation macro-F1. Test dùng báo cáo so sánh sau khi chọn.
Thời gian eval gồm đọc đặc trưng và truyền dữ liệu, không phải độ trễ realtime end-to-end.
Unknown dựa ngưỡng heuristic, cần bộ dữ liệu ngoài lớp để hiệu chỉnh và đánh giá thực tế.

## Tệp kết quả
- outputs/data: audit, metadata, thống kê, manifest train/val/test, nguồn trùng và mẫu bị purge.
- outputs/checkpoints: best.pt, last.pt, history.csv, validation predictions và trạng thái hoàn tất.
- outputs/metrics: accuracy, precision, recall, F1, confusion, ROC/AP, bảng so sánh và trọng số ensemble.
- outputs/figures: PNG/SVG để chèn báo cáo; outputs/predictions: xác suất từng mẫu và phân tích lỗi.
- configs: cấu hình, tên lớp, normalization, AST config/processor và nguồn pretrained.
- src/engine.py, scripts/predict.py: mã và lệnh dự đoán lại sau tải về.
- notebooks: snapshot mã đã chạy; notebook có đầy đủ output được lưu bằng Kaggle Save Version.
Cache đặc trưng phục vụ chạy tiếp, không đưa vào ZIP để tránh nhân đôi dung lượng.

## Tài liệu tham khảo
1. https://arxiv.org/abs/2209.04075
2. https://urbansounddataset.weebly.com/urbansound8k.html
3. https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer
4. https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
'''
    (root/'README.md').write_text(intro,encoding='utf-8')
    resulttext=results.to_string(index=False)
    (root/'outputs/report.md').write_text(intro+'\n## Kết quả đo được\n\n```text\n'+resulttext+'\n```\n',encoding='utf-8')
    html_doc='<html><meta charset="utf-8"><title>Báo cáo UrbanSound 7 lớp</title><body style="font-family:Arial;max-width:1200px;margin:auto;padding:30px">'
    html_doc+='<h1>Phân loại âm thanh đô thị — 7 lớp</h1><pre style="white-space:pre-wrap">'+html.escape(intro)+'</pre>'
    html_doc+=results.to_html(index=False,float_format=lambda x:f'{x:.4f}')+''.join(links)+'</body></html>'
    (root/'outputs/report.html').write_text(html_doc,encoding='utf-8')
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout
    (root/'outputs/logs/pip_freeze.txt').write_text(freeze,encoding='utf-8')
    # Executed input snapshot: no fabricated cell outputs. Kaggle saves actual output cells itself.
    try:
        import nbformat
        from IPython import get_ipython
        history=get_ipython().history_manager.input_hist_raw
        snap=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in history[1:] if s.strip()])
        snap.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
        nbformat.write(snap,root/'notebooks/executed_code_snapshot.ipynb')
    except Exception as e: print('Snapshot notebook:',e)
    manifest=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and 'cache' not in p.relative_to(root).parts:
            manifest.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size})
    pd.DataFrame(manifest).to_csv(root/'outputs/artifact_manifest.csv',index=False)
    if cfg['make_zip']:
        import zipfile
        dest=root.parent/(root.name+'.zip')
        with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for row in manifest:
                # Last optimizer states remain in Output; archive contains deployable best weights.
                if row['path'].endswith('/last.pt'): continue
                z.write(root/row['path'],arcname=root.name+'/'+row['path'])
            z.write(root/'outputs/artifact_manifest.csv',arcname=root.name+'/outputs/artifact_manifest.csv')
        display(FileLink(str(dest)))
    display(FileLink(str(root/'outputs/report.html'))); print('Output:',root)
    return root/'outputs/report.html'
