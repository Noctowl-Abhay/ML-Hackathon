import argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--root',default='Data')
a=p.parse_args()
root=Path(a.root)
if not root.exists(): raise SystemExit(f'Folder not found: {root}')
print('FILES (first 120):')
for f in list(sorted(x for x in root.rglob('*') if x.is_file()))[:120]: print(f, f'{f.stat().st_size/1024/1024:.2f} MB')
print('\nCANDIDATE TRAINING TABLES:')
found=[]
for base in [root,Path('outputs'),Path('data')]:
    if base.exists(): found.extend(f for f in base.rglob('*') if f.is_file() and 'features_train' in f.name and f.name.endswith(('.parquet','.csv','.csv.gz')))
for f in sorted(set(found)):
    print('\n',f)
    try:
        df=pd.read_parquet(f) if f.suffix=='.parquet' else pd.read_csv(f,nrows=5)
        print('Columns:',list(df.columns))
        print('Has trip_id,label,label_usable:',{'trip_id','label','label_usable'}.issubset(df.columns))
    except Exception as e: print('Cannot inspect:',e)
if not found: print('NONE FOUND. GPS-only preprocessing is not enough. Generate candidate-level features and aligned labels first.')
