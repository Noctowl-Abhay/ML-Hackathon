# PathSnap — Phase 1: dataset exploration

This starter contains data-inventory and Folium-visualisation examples. It does NOT include training data, model training, labels, or a finished solution. Please understand, adapt, and document your own experiments.

## Set up in Ubuntu Terminal

```bash
mkdir -p ~/PathSnap && cd ~/PathSnap
unzip -q ~/Downloads/'Student folder.zip' -d data
mv 'data/Student folder' data/raw
unzip -q ~/Downloads/PathSnap_student_materials.zip -d docs
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-phase1.txt
python src/inspect_dataset.py
python src/visualize_trip.py
xdg-open outputs/maps/00000005_exploration.html
```

Copy all the files from this starter archive into `~/PathSnap` before creating the virtual environment. If the supplied ZIPs are stored somewhere other than Downloads, substitute the correct paths.

Outputs:
- `outputs/results/dataset_inventory.csv`
- `outputs/maps/00000005_exploration.html`

### Expected check against supplied data
- train: 80 trips, 208417 GPS observations
- student_test: 10 trips, 19160 GPS observations

### Important constraints
- `.track` columns are longitude, latitude, timestamp (s).
- `.nodes` columns are longitude, latitude; node ID = 0-based row index.
- `.arcs` columns are from_node_id, to_node_id; arc ID = 0-based row index.
- `.route` is a chronological sequence of arc IDs, not a row-to-row GPS label array.
- Do not use evaluator data during development, and do not publish restricted raw data.
