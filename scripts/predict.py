from pathlib import Path
import sys, json, argparse
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from engine import predict_file
p=argparse.ArgumentParser(); p.add_argument('audio'); p.add_argument('--fold',type=int,default=None)
a=p.parse_args()
print(json.dumps(predict_file(a.audio,root,a.fold),ensure_ascii=False,indent=2))
