#!/usr/bin/env python3
"""Safely inspect teammate UI: identify entry points and existing API endpoints. No editing."""
from pathlib import Path
import re
root=Path('ui')
if not root.is_dir():raise SystemExit('ui/ not found; git fetch and merge latest main first.')
files=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.html','.js','.jsx','.ts','.tsx','.json'} and 'node_modules' not in p.parts]
print('UI source files:',len(files))
for p in sorted(files)[:100]:
    print('\n',p)
    content=p.read_text(encoding='utf-8',errors='replace')
    for line in content.splitlines():
        if re.search(r'fetch\(|axios|localhost|127\.0\.0\.1|/predict|/api/|/trips|mapbox|leaflet|openstreetmap',line,re.I):
            print('  ',line[:240])
print('\nNOTE: this is read-only; inspect which frontend functions need PathSnapAPI.listTrips()/predict().')
