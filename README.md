# ML-Hackathon
```
pathsnap/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── configs/
│   ├── data.yaml
│   ├── models.yaml
│   ├── bayes_opt.yaml
│   └── hmm.yaml
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── student_test/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   ├── projection.py
│   │   └── splits.py
│   │
│   ├── candidates/
│   │   ├── __init__.py
│   │   └── generate.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── extract.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   └── gradient_boosting.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── plots.py
│   │
│   ├── tuning/
│   │   ├── __init__.py
│   │   └── optuna_tuning.py
│   │
│   ├── hmm/
│   │   ├── __init__.py
│   │   ├── emission.py
│   │   ├── transition.py
│   │   └── viterbi.py
│   │
│   ├── viz/
│   │   ├── __init__.py
│   │   └── folium_map.py
│   │
│   └── pipeline.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_hmm_analysis.ipynb
│
├── results/
│   ├── logs/
│   ├── tables/
│   ├── figures/
│   ├── maps/
│   └── optuna/
│
├── artifacts/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── routes.py
│   └── evaluator.py
│
├── ui/
│   ├── app.py
│   └── requirements.txt
│
├── scripts/
│   ├── run_all.sh
│   └── export_artifacts.py
│
├── tests/
│   ├── test_candidates.py
│   ├── test_features.py
│   └── test_viterbi.py
│
└── report/
```
## 5. Assumptions Influencing the Proposed Map-Matching Approach

The following assumptions guide the proposed GPS trajectory map-matching approach. These are initial modelling assumptions that will be evaluated during implementation.

### 5.1 GPS Positional Noise
GPS observations may deviate from the actual road due to measurement errors, signal interference, or environmental conditions.

**Proposed approach:** Generate multiple nearby candidate road segments rather than assuming that every GPS point lies directly on a road.

### 5.2 Spatial Proximity
The correct road segment is assumed to be among the nearby candidates for most GPS observations.

**Proposed approach:** Use spatial indexing to identify the nearest road segments within a configurable search radius. Evaluate candidate recall to determine whether this assumption holds.

### 5.3 Temporal Continuity
Consecutive GPS observations generally represent continuous vehicle movement, except where significant timestamp gaps occur.

**Proposed approach:** Use timestamps, displacement, speed, and direction to assess movement between observations. Investigate segmenting trajectories when unusually large gaps occur.

### 5.4 Road-Network Connectivity
The reconstructed route should follow physically connected road segments and respect the road network's represented direction of travel.

**Proposed approach:** Use a road-network graph and transition probabilities to discourage disconnected or implausible movements.

### 5.5 Variable GPS Sampling Intervals
GPS observations are not necessarily recorded at uniform time intervals.

The dataset inspection identified:
- 7,286 training intervals exceeding one second.
- 264 student-testing intervals exceeding one second.
- A maximum observed interval of 153,064 seconds.

**Proposed approach:** Calculate movement features using actual timestamp differences rather than assuming a constant sampling interval.

### 5.6 Reference Route Alignment
The `.route` files contain ordered reference road-segment IDs, but the number of route entries differs from the number of GPS observations.

Therefore, reference route IDs cannot be assigned directly to GPS observations based on row position.

**Proposed approach:** Develop and validate a spatial and sequence-aware method for aligning GPS observations with reference route segments before supervised model training.

### 5.7 Geographical Distance Calculation
GPS coordinates are expressed as longitude and latitude, whereas candidate selection and movement analysis require meaningful physical distances.

**Proposed approach:** Use an appropriate coordinate projection or geodesic distance calculation to express distances in metres.

### 5.8 Road-Network Representation
The `.nodes` and `.arcs` files are assumed to describe the road network available for map matching.

However, the presence of reverse-direction arcs, road restrictions, and the completeness of the network must be investigated.

**Proposed approach:** Preserve the supplied arc directions and validate network connectivity before implementing route transitions.

### 5.9 Separation of Training and Evaluation Data
Reference routes are available for supervised learning and performance evaluation, but must not influence predictions on unseen trips.

**Proposed approach:** Train models using training trips, use group-aware validation, and reserve student-testing trips for development evaluation. Evaluator data will remain separate.

### 5.10 Sequential Map Matching
Predicting road segments independently may produce disconnected or unrealistic routes.

**Proposed approach:** Train five supervised candidate-scoring models—Logistic Regression, Distance-Weighted KNN, Random Forest, XGBoost, and Deep MLP—and investigate HMM/Viterbi decoding to improve route consistency.

The benefit of sequential decoding will be verified experimentally rather than assumed.

## Conclusion

The proposed PathSnap system assumes that GPS observations contain useful spatial and temporal information, that the supplied road network provides plausible route candidates, and that movement continuity can help resolve ambiguous road segments.

These assumptions will guide preprocessing, candidate generation, feature engineering, supervised learning, and HMM-based route reconstruction.

Their validity will be assessed using candidate coverage, observation-level accuracy, route continuity, and other evaluation metrics.
