import argparse,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--artifacts',default='artifacts')
a=p.parse_args()
rows=[]
for f in Path(a.artifacts).glob('*/metrics.json'):
    m=json.loads(f.read_text()); rows.append({k:v for k,v in m.items() if k not in ['features','validation_trip_ids']})
if not rows: raise SystemExit('No metrics found; train models first.')
df=pd.DataFrame(rows)
# Comparisons are only valid for identical validation trips and feature sets.
records=[json.loads(f.read_text()) for f in Path(a.artifacts).glob('*/metrics.json')]
if len({tuple(m['validation_trip_ids']) for m in records})>1: print('WARNING: validation trip sets differ; results are NOT directly comparable.')
if len({tuple(m['features']) for m in records})>1: print('WARNING: feature lists differ; results are NOT a controlled comparison.')
df=df.sort_values('top1_accuracy_when_positive_present' if 'top1_accuracy_when_positive_present' in df and df.top1_accuracy_when_positive_present.notna().any() else 'f1',ascending=False)
print(df.to_string(index=False))
out=Path(a.artifacts)/'model_comparison.csv';df.to_csv(out,index=False);print('Saved:',out)
