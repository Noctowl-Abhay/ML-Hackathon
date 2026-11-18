# PathSnap — Distributed GPS Map-Matching and Route Reconstruction

**PES University · UE24CS352A Machine Learning Hackathon · Technical Report Revision 2.0 (18–19 November 2026)**

> Team: _[Team member names and SRNs — complete before submission]_
> Repository: https://github.com/Noctowl-Abhay/ML-Hackathon

PathSnap takes noisy GPS observations (longitude, latitude, time) and a directed road network, decides which road segment each observation belongs to, and reconstructs a physically plausible vehicle route. It does this by turning map matching into **binary candidate classification (Logistic Regression) followed by sequential route decoding (HMM/Viterbi)**, and serves the result through a **FastAPI** backend and a browser UI that runs on a second laptop.

---

## Table of contents

1. [Headline results](#1-headline-results)
2. [Evidence conventions and current status](#2-evidence-conventions-and-current-status)
3. [Approach and architecture](#3-approach-and-architecture)
4. [Why Logistic Regression for a route problem?](#4-why-logistic-regression-for-a-route-problem)
5. [Dataset and exploratory analysis](#5-dataset-and-exploratory-analysis)
6. [Preprocessing and candidate generation](#6-preprocessing-and-candidate-generation)
7. [The 25 engineered features](#7-the-25-engineered-features)
8. [Model experimentation (five families)](#8-model-experimentation-five-families)
9. [Individual model summaries](#9-individual-model-summaries)
10. [Bayesian hyperparameter optimisation](#10-bayesian-hyperparameter-optimisation)
11. [HMM / Viterbi route reconstruction](#11-hmm--viterbi-route-reconstruction)
12. [Folium visualisation](#12-folium-visualisation)
13. [API, UI and deployment](#13-api-ui-and-deployment)
14. [How to run](#14-how-to-run)
15. [Modelling assumptions](#15-modelling-assumptions)
16. [Repository structure](#16-repository-structure)
17. [Limitations, integrity and remaining checks](#17-limitations-integrity-and-remaining-checks)
18. [Defending the design: likely evaluator questions](#18-defending-the-design-likely-evaluator-questions)
19. [Metric definitions](#19-metric-definitions)

---

## 1. Headline results

**Deployed scorer:** Logistic Regression refitted on all 80 training trips, grid-tuned (`C = 0.1`, `class_weight = None`), 25 features.

| Metric (10 student-test trips) | Value |
|---|---:|
| Candidate accuracy | 94.94% |
| Candidate macro-F1 | 91.03% |
| Conditional directed Top-1, classifier only | 85.92% |
| Conditional directed Top-1, **after HMM/Viterbi** | **90.53%** (+4.60 pp) |
| Route-switch continuity (before → after HMM) | 95.57% → **99.87%** |
| Disconnected arc switches (before → after HMM) | 258 → **5** |
| Mean ordered sequence F1 (before → after HMM) | 46.35% → **79.24%** |
| Candidate recall (reference arc present among candidates) | 99.59% (miss rate 0.41%) |
| Candidate-scoring time, 112,748 rows | 0.064 s |

**Final API contract check (10 reserved trips, local):** 10/10 responses successful, weighted directed Top-1 80.26% → **91.07%** after HMM over 19,491 eligible points, mean server latency 859.3 ms.

All accuracy-like values are measured against a **proxy reference alignment**, not survey-grade ground truth and not the organiser's official route score.

---

## 2. Evidence conventions and current status

| Designation | Meaning |
|---|---|
| RECOMPUTED — RAW DATA | Calculated by reading the raw student/training files |
| RECORDED — TERMINAL LOG | Saved CSV/JSON, model histories, trial logs, genuine screenshots (fitted weights not rerun when the report was written) |
| INSPECTED — SOURCE | Implementation details checked against source packages |
| PENDING | No verified evidence of completion |
| NOT AN OFFICIAL SCORE | Proxy-alignment metrics, not the organiser's route score |

**Outstanding before submission:**

- Physical two-laptop demo (Windows UI → Ubuntu API over LAN). Current evidence is **local loopback only**.
- Public repository verification (accessibility, no secrets, no restricted data).
- Evaluator-data integrity review (see [§17](#17-limitations-integrity-and-remaining-checks)).
- Team names/SRNs, contribution and AI-use disclosure.

---

## 3. Approach and architecture

PathSnap does **not** predict GPS coordinates. It predicts **which existing road arc** each GPS observation belongs to.

```text
              Raw GPS Observations
                       │
                       ▼
              Data Preprocessing
                       │
                       ▼
           Candidate Road Generation
          (UTM projection + STRtree, k ≤ 8, 50 m)
                       │
                       ▼
          25 Engineered Features per
              GPS–Candidate Pair
                       │
                       ▼
          LOGISTIC REGRESSION CLASSIFIER
                       │
                       ▼
           Candidate Matching Scores
                       │
                       ▼
          HMM + VITERBI SEQUENCE DECODER
                       │
                       ▼
           Connected Road Reconstruction
                       │
                       ▼
                  FastAPI  (Laptop 1, Ubuntu)
                       │  HTTP
                       ▼
          PathSnap Frontend (Laptop 2, Windows)
```

| Component | Responsibility |
|---|---|
| Candidate generator | Finds nearby directed road arcs for each GPS point |
| Logistic Regression | Candidate classification and probability estimation |
| HMM-style model | Combines classifier (emission) scores with transition scores |
| Viterbi | Finds the most plausible sequence of road candidates |
| Road-network graph | Directed connectivity and path lengths |
| Folium | Geographical route visualisation (offline report use) |
| FastAPI | Serves predictions; evaluates privately server-side |

### Research questions and evaluation hierarchy

1. **Candidate classification** – can the classifier tell a reference-aligned arc from alternatives?
2. **Observation-level matching** – is the top-ranked candidate the aligned reference arc?
3. **Sequence reconstruction** – does HMM/Viterbi improve selection while reducing disconnected transitions?
4. **Distributed system** – can a Windows UI request live trips from an Ubuntu API that holds the model, road data and private labels, without labels returning to the UI?

Primary development metric: **conditional directed observation-level Top-1**, supported by macro-F1, log loss, route quality, CV stability and inference cost. Candidate accuracy alone is insufficient because negatives outnumber positives.

---

## 4. Why Logistic Regression for a route problem?

Despite the name, Logistic Regression is a **supervised classification** algorithm that estimates the probability that an observation belongs to a class. We convert map matching into a **binary classification problem followed by sequential route reconstruction**.

### 4.1 What exactly are we predicting?

For every GPS observation we generate nearby candidate road arcs. For each candidate, the model answers:

> **Is this candidate road arc the correct match for this GPS observation?**

| Class | Meaning |
|---|---|
| `1` | Correct candidate road arc |
| `0` | Incorrect candidate road arc |

The target is **categorical**, so classification is the appropriate formulation.

### 4.2 How it works, step by step

**Step 1 – Candidate generation.** Search for road arcs within the configured radius (50 m) and keep up to eight.

**Step 2 – Feature extraction.** Each GPS–candidate pair becomes one classification sample with 25 features. Illustrative example:

| Feature | Arc A | Arc B | Arc C | Arc D |
|---|---:|---:|---:|---:|
| Distance from GPS | 3 m | 6 m | 12 m | 20 m |
| Heading difference | 8° | 170° | 45° | 90° |
| Candidate rank | 1 | 2 | 3 | 4 |
| Arc length | 120 m | 120 m | 80 m | 60 m |
| Remaining features | ... | ... | ... | ... |
| Training label | **1** | 0 | 0 | 0 |

**Step 3 – Probability.** The model forms a weighted sum of the features and applies the sigmoid:

$$z = b + w_1 x_1 + w_2 x_2 + \cdots + w_{25} x_{25}$$

$$P(y=1 \mid X) = \frac{1}{1 + e^{-z}}$$

where $x_1,\ldots,x_{25}$ are the engineered features, $w_i$ are learned weights and $b$ is the intercept.

**Step 4 – Rank the candidates.** Illustrative scores:

| Candidate | Probability of correct match | Rank |
|---|---:|---:|
| Arc A | **0.94** | 1 |
| Arc C | 0.31 | 2 |
| Arc B | 0.18 | 3 |
| Arc D | 0.04 | 4 |

The classifier-only choice is Arc A:

$$\hat{a}_t = \arg\max_{a \in C_t} P(y=1 \mid X_{t,a})$$

The 0.5 threshold is used for the reported binary metrics (accuracy, macro-F1, confusion matrices), but for map matching the probabilities are mainly used to **rank competing candidates**. The independent candidate probabilities do not need to sum to 1.

### 4.3 Why Logistic Regression alone is not enough

| GPS point | Candidate A | Candidate B |
|---|---:|---:|
| GPS 1 | **0.95** | 0.10 |
| GPS 2 | **0.92** | 0.15 |
| GPS 3 | 0.65 | **0.72** |
| GPS 4 | **0.94** | 0.12 |
| GPS 5 | **0.97** | 0.08 |

Picking the highest score independently gives `A → A → B → A → A`. Road B may not connect to Road A consistently with the observed movement; a vehicle cannot teleport. **Logistic Regression scores individual candidates but does not enforce global route connectivity.**

HMM/Viterbi uses both:

- **Emission-like scores** – how plausible each candidate is according to the classifier.
- **Transition scores** – how plausible movement between consecutive candidates is according to the road network and observed displacement.

```text
Classifier-only prediction:

    A ─── A     B     A ─── A
                ↑
         Possible discontinuity

HMM/Viterbi prediction:

    A ─── A ─── A ─── A ─── A
         Connected sequence
```

Even though B scored higher at GPS 3, A can be more plausible over the whole trajectory.

### 4.4 Why not Linear Regression?

| Linear Regression | Logistic Regression |
|---|---|
| Predicts continuous values | Predicts class probabilities |
| Output can be any real number | Probability between 0 and 1 |
| Suited to temperature, price, distance | Suited to correct/incorrect classification |
| Squared-error loss | Binary cross-entropy loss |
| Not our formulation | **Appropriate for candidate classification** |

Road arc IDs are identifiers, not meaningful quantities: if arcs 100 and 200 are candidates, "arc 150" means nothing.

### 4.5 Why we chose it over the other four families

| Model | Candidate accuracy | Macro-F1 | Top-1 after HMM | Candidate-scoring time |
|---|---:|---:|---:|---:|
| **Tuned Logistic Regression (full-80)** | 94.94% | 91.03% | 90.53% | **0.064 s** |
| Weighted KNN | 94.42% | 90.10% | 87.89% | 156.405 s |
| Random Forest | 94.49% | 90.70% | **91.36%** | 0.865 s |
| XGBoost | **95.06%** | **91.11%** | 90.69% | 0.121 s |
| Deep MLP | 94.60% | 90.22% | 89.34% | 1.253 s |

> **Fairness note:** the four non-LR rows are the original saved fits trained on 64 trips; the tuned LR is refitted on all 80 training trips. This is informative but not a strict equal-training-size contest.

Reasons:

- **A. Efficiency** – fastest recorded scoring time, good for interactive reconstruction.
- **B. Competitive performance** – accuracy and macro-F1 close to far more complex models.
- **C. Probability estimates** – directly usable as HMM observation scores.
- **D. Interpretability** – coefficients show how distance, direction, motion and topology features contribute.
- **E. Well-engineered features** – the 25 features already encode geography and movement, so the model need not learn them from raw coordinates.
- **F. Regularisation** – final model uses `C = 0.1` to control complexity.

**Logistic Regression is not best on every metric.** Random Forest gives higher post-HMM Top-1 (+0.83 pp) and XGBoost gives better macro-F1 and log loss. LR was chosen as a balance of accuracy, speed and deployment suitability, not because it wins everything.

---

## 5. Dataset and exploratory analysis

### 5.1 File schema

| Extension | Contents | Interpretation |
|---|---|---|
| `.track` | longitude, latitude, timestamp | Observed GPS sequence, no header |
| `.nodes` | longitude, latitude | Road vertices; row index = node ID |
| `.arcs` | from_node_id, to_node_id | Directed road segment; row index = arc ID |
| `.route` | ordered integer arc IDs | Reference route; **length differs from GPS count** |

Partitions: **80 training**, **10 student-test**, **10 evaluator-test** trips. The evaluator split must stay excluded from development, feature selection and tuning.

### 5.2 Recomputed inventory (RECOMPUTED — RAW DATA)

| Partition | Trips | GPS points | Road nodes | Directed arcs | Reference arcs | Gaps > 1 s |
|---|---:|---:|---:|---:|---:|---:|
| Training | 80 | 208,417 | 1,933,485 | 3,876,904 | 31,336 | 7,286 |
| Student test | 10 | 19,160 | 90,689 | 185,041 | 3,477 | 264 |
| **Development total** | **90** | **227,577** | **2,024,174** | **4,061,945** | **34,813** | **7,550** |

Data quality: all 227,577 development GPS rows are finite and in range; zero duplicate (lon, lat, time) rows; all arc endpoint IDs and reference route IDs in range; zero isolated nodes across the 90 road networks (this does **not** imply the networks are fully connected).

- Training trajectories: 361–19,313 observations (median 1,529); path length ≈ 6.33–169.27 km.
- Student-test: 262–3,958 observations (median 1,420); path length ≈ 5.55–71.72 km.
- Largest observed time gap: 153,064 s.

### 5.3 Student-test trip inventory

| Trip ID | GPS | Nodes | Arcs | Gaps > 1 s | Weak components |
|---|---:|---:|---:|---:|---:|
| 00000000 | 2,503 | 8,542 | 18,196 | 1 | 100 |
| 00000007 | 1,354 | 2,061 | 4,369 | 0 | 6 |
| 00000009 | 1,049 | 6,745 | 13,106 | 1 | 8 |
| 00000040 | 2,995 | 12,945 | 27,026 | 0 | 15 |
| 00000052 | 262 | 1,770 | 3,577 | 4 | 7 |
| 00000056 | 3,958 | 14,759 | 27,631 | 41 | 24 |
| 00000065 | 768 | 6,448 | 13,184 | 11 | 5 |
| 00000076 | 1,486 | 26,635 | 56,027 | 206 | 40 |
| 00000077 | 3,911 | 4,424 | 9,622 | 0 | 11 |
| 00000091 | 874 | 6,360 | 12,303 | 0 | 11 |

All ten road graphs have **multiple weakly connected components**, so decoder transitions must respect **directed accessibility**, not just Euclidean closeness. Example: trip 00000000 has 100 components; its largest has 7,466 nodes.

### 5.4 Data-processing decisions

- Preserve raw observations; create processed files and flags instead of overwriting.
- Check coordinate validity, time monotonicity, duplicates and gaps.
- Treat `.route` as an ordered arc sequence, **not** one label per GPS row.
- Project to a local metric CRS before applying 30–50 m thresholds.
- Keep training-only learned transforms and reference-derived labels separate from inference input.

---

## 6. Preprocessing and candidate generation

- GPS is projected to a **local UTM metric CRS**; directed road arcs become line segments in a **Shapely STRtree**.
- Per GPS point: **at most k = 8** nearest arcs **within 50 m**. If the radius query is empty, the nearest arc is kept as a **fallback** (`is_fallback`). No separate k/radius optimisation study is claimed.
- The ordered reference `.route` is aligned to GPS observations by **monotonic dynamic programming**. A candidate matching the aligned reference arc gets label 1, others 0. The label-distance cutoff is **30 m**; labels beyond it may be marked unusable. This alignment is a **proxy** target.
- Reference-only fields (`label_arc_id`, `label_route_pos`, `label_dist_m`, `label_usable`, `point_has_positive`) **never** enter the 25-column input matrix.
- Cross-validation is always grouped by `trip_id`.

**Candidate measurements**

| Quantity | Value |
|---|---:|
| Training candidate rows (original artifact) | 1,102,890 (1,037,041 usable) |
| Student-test usable candidate rows | 112,748 |
| Student eligible observations (unique-positive condition) | 18,969 |
| Student GPS points / reference-missing candidate sets | 19,160 / 79 |
| Candidate recall / miss rate | 99.59% / 0.41% |
| Points excluded from conditional Top-1 | 191 (not identical to candidate-miss count) |

---

## 7. The 25 engineered features

The design principle is **complementarity**: distance gives proximity; heading gives direction; motion and neighbour intervals expose local ambiguity; topology gives feasible transitions; flags stop unreliable values being treated as ordinary numbers.

**Geometric (10)**

| Feature | Reason |
|---|---|
| `dist_m` | Point-to-arc separation in metres |
| `cand_rank` | Distance order among candidates |
| `dist_minus_min_m` | Margin from nearest arc (relative ambiguity) |
| `dist_ratio_min` | Scale-adjusted distance ratio |
| `n_cands_in_radius` | Local road density near junctions |
| `frac_along` | Projection fraction along the directed arc |
| `frac_to_end` | Closeness to segment endpoints |
| `arc_length_m` | Arc length, relevant to movement plausibility |
| `endpoint_dist_m` | Distance to endpoint (intersection choices) |
| `projection_outside` | Projection beyond segment ends (discourages implausible matches) |

**Heading (3)**

| Feature | Reason |
|---|---|
| `heading_diff_deg` | Circular difference between GPS heading and arc bearing (0–180°) |
| `heading_diff_undirected_deg` | Mismatch ignoring reverse polarity |
| `heading_valid` | Whether motion supports a meaningful heading (e.g. not stopped) |

**Motion (8)**

| Feature | Reason |
|---|---|
| `speed_mps` | Travel-feasibility context |
| `gps_step_m` | Displacement between successive observations |
| `dt_s` | Elapsed time since previous observation |
| `prev_step_m`, `next_step_m` | Previous/following displacement |
| `prev_dt_s`, `next_dt_s` | Past/upcoming sampling gaps |
| `movement_valid` | Reliability of displacement-derived motion |

**Road (3)**

| Feature | Reason |
|---|---|
| `arc_bearing_deg` | Arc orientation, combined with heading |
| `degree_from`, `degree_to` | Graph degree at start/end node (branching) |

**Fallback (1):** `is_fallback` – forced nearest-arc choice when radius search was empty.

---

## 8. Model experimentation (five families)

### 8.1 Protocol

- Original five pipelines: **64/16 trip-disjoint split** of the 80 training trips (seed 42).
- Imputation and scaling are **fitted inside each training pipeline**: median imputation for all; `StandardScaler` for LR, KNN and MLP; no scaling for tree ensembles.
- **Three-fold GroupKFold by `trip_id`** on all 80 training trips for CV.
- Student-test results evaluate the saved models on the separate 10 trips. The untuned, grid-tuned and Bayesian LR variants are refitted on the **full 80** trips, so training size differs from the original five.
- Candidate threshold 0.5. Conditional Top-1 selects one highest-scoring candidate per eligible GPS point.

### 8.2 Original 64/16 holdout (211,131 validation candidate rows)

| Model | Accuracy | Precision | Recall | Binary F1 | Macro-F1 | Top-1 | Log loss | Fit (s) | Infer (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 91.37% | 69.90% | 90.71% | 78.95% | 86.76% | 84.08% | 0.2262 | 7.782 | 0.257 |
| Weighted KNN | 91.84% | 77.21% | 77.00% | 77.10% | 86.07% | 78.14% | 0.5733 | 3.202 | 256.228 |
| Random forest | 91.18% | 72.69% | 80.99% | 76.62% | 85.59% | 77.73% | 0.2032 | 248.683 | 1.668 |
| XGBoost | 91.49% | 77.83% | 73.16% | 75.42% | 85.14% | 76.36% | 0.2182 | 43.560 | 0.465 |
| Deep MLP | 90.84% | 76.20% | 70.75% | 73.37% | 83.92% | 75.19% | 0.3800 | 1750.266 | 2.600 |

LR led this holdout on macro-F1 and Top-1, which promoted it for further study.

### 8.3 Mandatory five-model comparison

| Model | Key configuration | CV macro-F1 | Student Top-1 | Route continuity (post-HMM) | Student macro-F1 | Log loss | Infer (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | C=1; balanced | 87.38% | 85.64% | 99.84% | 87.99% | 0.2079 | 0.106 |
| Weighted KNN | k=7; distance | 87.99% | 84.26% | 99.73% | 90.10% | 0.3190 | 156.405 |
| Random forest | 200 trees; depth 18 | **89.36%** | **85.92%** | 99.84% | 90.70% | 0.1410 | 0.865 |
| XGBoost | 350 trees; depth 5 | 89.12% | 85.49% | 99.84% | **91.11%** | **0.1221** | 0.121 |
| Deep MLP | 128/64/32; ReLU | 87.40% | 84.99% | 99.81% | 90.22% | 0.1419 | 1.253 |

### 8.4 All student-test classifier results (five families + three LR variants)

| Model | Accuracy | Precision | Recall | Binary F1 | Macro-F1 | Top-1 | Log loss | Infer (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 92.49% | 71.22% | 92.95% | 80.65% | 87.99% | 85.64% | 0.2079 | 0.106 |
| Weighted KNN | 94.42% | 82.81% | 84.33% | 83.56% | 90.10% | 84.26% | 0.3190 | 156.405 |
| Random forest | 94.49% | 79.25% | 91.10% | 84.76% | 90.70% | 85.92% | 0.1410 | 0.865 |
| XGBoost | 95.06% | 86.01% | 84.38% | 85.19% | 91.11% | 85.49% | 0.1221 | 0.121 |
| Deep MLP | 94.60% | 85.05% | 82.36% | 83.68% | 90.22% | 84.99% | 0.1419 | 1.253 |
| LR untuned (80) | 92.45% | 71.01% | 93.12% | 80.57% | 87.94% | 85.85% | 0.2095 | 0.066 |
| **LR grid (80) — deployed** | 94.94% | 84.29% | 85.93% | 85.10% | 91.03% | 85.92% | 0.1431 | 0.064 |
| LR Bayesian (80) | 94.94% | 84.30% | 85.92% | 85.11% | 91.03% | 85.91% | 0.1431 | 0.068 |

### 8.5 Three-fold grouped CV (macro-F1)

| Model | Fold 1 | Fold 2 | Fold 3 | Mean | SD | Mean loss | Fit Σ (s) | Infer Σ (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 87.56% | 86.40% | 88.18% | 87.38% | 0.74 pp | 0.2172 | 15.438 | 0.670 |
| Weighted KNN | 87.09% | 87.34% | 89.54% | 87.99% | 1.10 pp | 0.4396 | 5.512 | 1412.065 |
| Random forest | 87.24% | 88.99% | 91.84% | 89.36% | 1.89 pp | 0.1559 | 450.509 | 6.041 |
| XGBoost | 86.64% | 88.80% | 91.91% | 89.12% | 2.16 pp | 0.1546 | 30.732 | 1.406 |
| Deep MLP | 85.61% | 87.78% | 88.80% | 87.40% | 1.33 pp | 0.2553 | 4120.241 | 9.115 |

The 64/16 holdout and the grouped CV are different experiments; their means cannot be substituted for student-test metrics.

### 8.6 Selection rule

The deployment manifest selects **grid-tuned full-80 LR (`C=0.1`, `class_weight=None`)** on the **student-test** partition. Its Top-1 (85.92%) ties Random Forest, its scoring time (0.064 s) is about **13.6×** faster than the original RF (0.865 s), and its macro-F1 slightly exceeds RF. XGBoost remains better on classification macro-F1 and log loss, and RF is 0.83 pp higher after HMM. The decision prioritises **fast, simple, stable live scoring with competitive Top-1**, not a claim that LR maximises every route measure.

---

## 9. Individual model summaries

| Model | Configuration | Strengths | Limitations |
|---|---|---|---|
| **Logistic Regression** | C=1.0, balanced, lbfgs, max_iter=1500, seed 42; median impute + standardise; L2 cross-entropy | Fast, probabilistic, interpretable; scores feed the HMM directly | No built-in feature interactions; lower precision when balanced |
| **Weighted KNN** | k=7, distance weights, euclidean, leaf_size=40, n_jobs=4 | Captures local nonlinear patterns, no parametric fit | Inference 156.405 s per student batch; highest log loss |
| **Random Forest** | 200 trees, depth 18, min_samples_leaf=3, sqrt features, balanced_subsample, gini | Highest CV macro-F1 (89.36%) and post-HMM Top-1 (91.36%) | Larger weights, slower than LR (0.865 s) |
| **XGBoost** | 350 trees, depth 5, lr 0.05, subsample/colsample 0.85, hist | Best student accuracy, macro-F1 and log loss; fast scoring | Less interpretable; largest CV fold SD (2.16 pp) |
| **Deep MLP** | (128,64,32), ReLU, adam, alpha 0.001, batch 256, early stopping (15% val, patience 10) | Flexible nonlinear scorer | 1,750 s fit, 4,120 s summed CV fit; no deployment advantage |

**Student-test confusion matrices** (rows = proxy reference, columns = predicted; 112,748 rows):

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic regression | 86,653 | 7,126 | 1,337 | 17,632 |
| Weighted KNN | 90,458 | 3,321 | 2,972 | 15,997 |
| Random forest | 89,254 | 4,525 | 1,688 | 17,281 |
| XGBoost | 91,175 | 2,604 | 2,963 | 16,006 |
| Deep MLP | 91,032 | 2,747 | 3,347 | 15,622 |
| LR untuned (80) | 86,567 | 7,212 | 1,305 | 17,664 |
| LR grid (80) | 90,742 | 3,037 | 2,669 | 16,300 |
| LR Bayesian (80) | 90,744 | 3,035 | 2,670 | 16,299 |

**Over/underfitting:** student macro-F1 exceeds the CV means for all five, so no held-out collapse is visible. Training-set metrics, depth/`k` ablations and boosting histories were not exported, so no definitive over/underfit claim is made. Only the MLP has an epoch-wise history (training loss plus an accuracy-like internal validation score, **not** a validation-loss curve); no curves are fabricated for the other models.

**KNN note:** the reconstruction script's separately scored baseline Top-1 is 84.08% versus 84.26% in the batched classifier comparison. The 0.18 pp source of difference is not established.

---

## 10. Bayesian hyperparameter optimisation

Optuna **TPE**, seed 42, on the **80 training trips only**. Each trial refits across the same **five trip-grouped folds**; the objective is mean conditional Top-1 (maximise).

| Search field | Recorded setting |
|---|---|
| `C` | Log-uniform [0.001, 100] |
| `class_weight` | `None` or `balanced` |
| Trials | 24 of 24 complete |
| Best trial | Index 10 (11th): `C = 0.05939820799766786`, `class_weight = None` |
| Best mean CV Top-1 | 85.05% |
| Search wall time | 531.38 s (8.86 min); summed trial fitting 502.94 s |
| Best-model OOF refit / final fit | 22.35 s / 4.53 s |

### Tuned vs untuned (matched folds and student partition)

| Metric | Untuned full-80 LR | Bayesian full-80 LR |
|---|---:|---:|
| Configuration | C=1; balanced | C=0.059398; None |
| 5-fold mean CV macro-F1 | 87.55% | 90.17% |
| 5-fold mean CV Top-1 | 85.02% | 85.05% |
| 5-fold mean CV log loss | 0.2159 | 0.1559 |
| Student candidate accuracy | 92.45% | 94.94% |
| Student macro-F1 | 87.94% | 91.03% |
| Student conditional Top-1 | 85.85% | 85.91% |
| Student log loss | 0.2095 | 0.1431 |
| Student batch inference (s) | 0.066 | 0.068 |

Tuning improves CV macro-F1 by **2.62 pp** and student macro-F1 by **3.09 pp**, but Top-1 only by 0.06 pp: ranking improves far less than thresholded classification. No latency improvement is claimed.

Bayesian pooled OOF: accuracy 94.38%, macro-F1 90.22%, binary F1 83.84%, Top-1 85.12%, log loss 0.1560; TN=827,432, FP=28,042, FN=30,269, TP=151,298. These OOF folds informed tuning, so this is not nested CV.

### Grid-tuned LR (the deployed one)

A separate five-fold grouped grid search chose `C=0.1`, `class_weight=None` (mean CV Top-1 85.05%, log loss 0.1559, pooled OOF accuracy 94.37%, pooled Top-1 85.12%). It is documented as a **distinct tuning approach, not mislabelled as Bayesian**. Bayesian and grid differ by 0.001 pp macro-F1 and 0.016 pp Top-1 (three eligible observations), so neither is claimed robustly superior without repeated trials.

Deployment identity: `artifacts/logistic_regression_tuned/model_all_80.joblib → deployment/selected_model.joblib`, 25 features, manifest SHA-256 `3f10a93b23c13584a2a60ae93edb7634c749e07133dfa98317b26669a921b0ad` (recorded from the manifest, **not independently rehashed**).

<details>
<summary>All 24 completed trials (zero-based index)</summary>

| Trial | C | Weight | Mean CV Top-1 | Mean CV loss | Fit (s) |
|---:|---:|---|---:|---:|---:|
| 0 | 0.074593433 | none | 85.05% | 0.1559 | 21.41 |
| 1 | 0.98467389 | none | 85.05% | 0.1560 | 21.54 |
| 2 | 0.0019517225 | none | 85.00% | 0.1566 | 17.93 |
| 3 | 3.470267 | balanced | 85.02% | 0.2159 | 21.88 |
| 4 | 14.528247 | none | 85.05% | 0.1560 | 20.81 |
| 5 | 0.0082608084 | balanced | 85.00% | 0.2160 | 21.98 |
| 6 | 0.14445251 | balanced | 85.02% | 0.2159 | 22.57 |
| 7 | 0.0049827524 | balanced | 85.00% | 0.2161 | 21.71 |
| 8 | 0.71198331 | none | 85.05% | 0.1560 | 21.87 |
| 9 | 70.418303 | balanced | 85.02% | 0.2159 | 22.78 |
| **10** | **0.059398208** | **none** | **85.05%** | **0.1559** | **21.85** |
| 11 | 0.030322664 | none | 85.05% | 0.1560 | 19.64 |
| 12 | 0.030225755 | none | 85.05% | 0.1560 | 21.11 |
| 13 | 0.014494925 | none | 85.04% | 0.1560 | 20.46 |
| 14 | 0.21171139 | none | 85.05% | 0.1559 | 20.73 |
| 15 | 0.0054344542 | none | 85.03% | 0.1561 | 20.09 |
| 16 | 0.14565833 | none | 85.05% | 0.1559 | 20.95 |
| 17 | 0.040826304 | balanced | 85.02% | 0.2160 | 20.37 |
| 18 | 0.93787334 | balanced | 85.02% | 0.2159 | 21.10 |
| 19 | 0.0011762385 | none | 84.96% | 0.1570 | 17.70 |
| 20 | 0.016439471 | none | 85.05% | 0.1560 | 21.33 |
| 21 | 0.030308354 | none | 85.05% | 0.1560 | 19.57 |
| 22 | 0.0591811 | none | 85.05% | 0.1559 | 21.50 |
| 23 | 0.28922157 | none | 85.05% | 0.1559 | 22.06 |

</details>

---

## 11. HMM / Viterbi route reconstruction

### 11.1 States and emissions

For each ordered GPS observation $t$, the hidden state is one directed road-arc candidate $a$ from its candidate set. The chosen classifier provides $p_t(a)$ from the same 25 inference features; **no reference label enters the score**.

$$\text{emission}_t(a) = w_e \log\big(\text{clip}(p_t(a),\, 10^{-8},\, 1-10^{-8})\big)$$

These are **discriminative emission potentials**, not independently fitted generative GPS-noise probabilities.

### 11.2 Road-aware transitions and recurrence

For adjacent observations the decoder computes travel along the old arc, a directed shortest path to the new arc, and distance along the new arc. Same-arc forward progress uses the fractional distance; backward motion within 7 m is allowed with a small penalty; U-turns between reversed endpoints get an extra penalty; a missing bounded connection gets a finite disconnection penalty.

With $d(i,j)$ the road-network travel estimate, $g_t$ the observed displacement, $\beta = 30$ m and $P(i,j)$ the penalties:

$$D_1(j) = w_e \log p_1(j)$$

$$D_t(j) = w_e \log p_t(j) + \max_i \Big[ D_{t-1}(i) - \frac{|d(i,j) - g_t|}{\beta} - P(i,j) \Big]$$

Predecessors are stored, each step's scores are shifted by a common maximum for numerical stability, and the best terminal state is backtracked. Complexity is $O(Tk^2)$ pairwise comparisons plus bounded graph-search work.

| Decoder setting | Value |
|---|---|
| Emission weight | 1.0 |
| Distance scale β | 30 m |
| Disconnected / U-turn penalties | 12.0 / 4.0 (log-score units) |
| Gap reset | Δt > 30 s, non-positive/invalid Δt, missing consecutive index or invalid displacement |
| Bounded transition search | min(500 m, max(100 m, 3 × observed displacement + 50 m)) |
| Backslide tolerance | 7 m |
| Post-selection path expansion | Directed connectors up to 1,000 m joining adjacent selected arcs |

No decoder-weight tuning study is claimed.

### 11.3 Gaps, missing candidates, invalid transitions

- Empty radius search → nearest-arc fallback so each valid point has a state set.
- Empty candidate lists, duplicate point/arc IDs or invalid arc indices are reported as errors.
- Large gaps or invalid observations **split the sequence** into independently decoded segments; no transition penalty crosses a segment boundary.
- A missing directed path receives the penalty and may remain broken.
- After Viterbi, connecting paths are expanded with directed shortest paths; connected/disconnected switches are counted explicitly.

### 11.4 Student-test: five families before/after HMM

| Model | Before | After | Change | Continuity | Broken | Mean seq. F1 | Elapsed (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 85.64% | 90.50% | +4.85 pp | 99.84% | 6 | **82.48%** | 4.871 |
| Weighted KNN | 84.08% | 87.89% | +3.81 pp | 99.73% | 11 | 72.97% | 1374.150 |
| Random forest | 85.92% | **91.36%** | +5.44 pp | 99.84% | 6 | 82.19% | 6.142 |
| XGBoost | 85.49% | 90.69% | +5.20 pp | 99.84% | 6 | 80.97% | 4.926 |
| Deep MLP | 84.99% | 89.34% | +4.34 pp | 99.81% | 7 | 80.55% | 6.902 |

All five improve with HMM. RF leads point-level decoding; the original LR has the best mean sequence F1, showing sequence quality is not determined by point accuracy alone.

### 11.5 Selected grid-tuned LR: paired route comparison

| Metric | Classifier-only + expansion | Classifier + HMM + expansion | Change |
|---|---:|---:|---|
| Directed Top-1 (weighted) | 85.92% | **90.53%** | +4.60 pp |
| Undirected Top-1 (weighted) | 91.97% | 94.89% | physical-segment proxy |
| Route-switch continuity | 95.57% | **99.87%** | +4.30 pp |
| Connected arc switches | 5,560 | 3,841 | different switch counts |
| Disconnected arc switches | 258 | **5** | −253 |
| Mean ordered sequence F1 | 46.35% | **79.24%** | +32.89 pp |
| Expanded arc occurrences | 12,108 | 5,311 | fewer repeated/detour arcs |

Re-expanded saved HMM paths matched the supplied per-trip files exactly. This is a post-prediction diagnostic only; it does not retrain, tune or touch evaluator labels.

### 11.6 Per-trip results (selected LR, 18,969 eligible points)

| Trip ID | Eligible | Before Top-1 | After Top-1 | Before breaks | After breaks | Before seq. F1 | After seq. F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00000000 | 2,496 | 93.95% | 95.51% | 15 | 2 | 40.27% | 74.41% |
| 00000007 | 1,327 | 93.37% | 97.06% | 2 | 0 | 66.30% | 95.28% |
| 00000009 | 1,018 | 87.52% | 89.78% | 0 | 0 | 64.95% | 85.63% |
| 00000040 | 2,946 | 92.70% | 95.62% | 13 | 0 | 45.35% | 80.05% |
| 00000052 | 262 | 95.04% | 100.00% | 5 | 0 | 67.64% | 98.94% |
| 00000056 | 3,919 | 89.13% | 95.89% | 202 | 2 | 54.34% | 89.90% |
| 00000065 | 767 | 82.14% | 86.70% | 6 | 1 | 28.77% | 79.67% |
| 00000076 | 1,472 | 87.91% | 91.71% | 1 | 0 | 44.51% | 76.33% |
| 00000077 | 3,888 | 69.26% | 75.82% | 0 | 0 | 22.22% | 45.08% |
| 00000091 | 874 | 83.98% | 89.93% | 14 | 0 | 29.19% | 67.15% |

**Failure cases:** the biggest break reduction is trip 00000056 (202 → 2). Trip 00000052 reaches 100% Top-1 but sequence F1 is 98.94%, not a perfect route. The hardest trip is **00000077** (75.82% Top-1, 45.08% sequence F1, despite zero breaks). Trips 00000000 and 00000065 retain two and one disconnected switches. HMM **reduces but does not eliminate** failures, and connectivity alone does not prove correctness.

---

## 12. Folium visualisation

`scripts/render_folium_guidelines.py` builds the interactive map from the processed GPS track, directed network and saved HMM route geometry. Layers: **raw GPS (blue)**, **road-network sample (gray)**, **predicted HMM/Viterbi route (green)**, **student-test reference (orange, report only)**; start/end markers, scale control and layer toggles. The map is tile-free to avoid a blocked basemap. The reference overlay is **never** enabled in the evaluator-facing UI.

Interactive output: `artifacts/reconstruction/student_test/00000000/road_network_map.html`. A visually overlapping route does not establish exact directed-arc accuracy; use the quantitative metrics above.

---

## 13. API, UI and deployment

```text
Laptop 2 (Windows): UI  ──HTTP──►  Laptop 1 (Ubuntu): FastAPI
                                      │ trip_id
                                      ▼
                          Live pipeline: GPS + road graph
                          → 25 features → LR score → HMM/Viterbi
                                      │
                                      ▼
                       Private server-side evaluation
                       (aggregate metrics only, no labels returned)
                                      │
                                      ▼
                       JSON: coordinates + metrics + latency_ms
```

| Endpoint / component | Contract |
|---|---|
| `GET /health` | Model, features and data-root availability |
| `GET /trips` (and `/api/trips`) | Ten trip IDs |
| `POST /predict` | Validated numeric `trip_id`; live reconstruction; returns `trip_id`, `gps`, `predicted_route`, `metrics`, `latency_ms` |
| `GET /api/reconstruct/{trip_id}` | Original reconstruction response with metrics, stats, GeoJSON |
| UI | Static browser app on port 5500; user enters the API URL, selects a trip, toggles observations and predicted route |
| Private evaluation | Only aggregate proxy Top-1, eligible coverage, sequence F1, unique-arc coverage, switch counts and latency are exposed |

The adapter converts `raw_gps` and `predicted_route` GeoJSON from lon/lat to lat/lon, validates shape/finiteness/range, rejects empty geometry and omits every other layer. Map display caps GPS samples at 4,000 (this does not change the prediction or eligible-point denominator). UI "Predicted segments" counts polylines, not individual arcs.

### Final frozen-model API contract results (10 reserved trips, local)

| Trip ID | Eligible | Before Top-1 | After Top-1 | Seq. F1 | Broken | API ms | Contract |
|---|---:|---:|---:|---:|---:|---:|---|
| 00000015 | 6,749 | 59.12% | 82.99% | 21.07% | 0 | 1996 | PASS |
| 00000017 | 1,320 | 91.29% | 94.77% | 87.13% | 0 | 950 | PASS |
| 00000023 | 454 | 94.05% | 95.15% | 65.92% | 0 | 218 | PASS |
| 00000030 | 629 | 81.24% | 87.28% | 83.23% | 0 | 391 | PASS |
| 00000049 | 2,640 | 97.69% | 99.17% | 96.58% | 0 | 1223 | PASS |
| 00000059 | 1,028 | 94.55% | 96.40% | 88.78% | 0 | 736 | PASS |
| 00000067 | 916 | 89.85% | 94.98% | 96.47% | 0 | 355 | PASS |
| 00000073 | 1,140 | 96.32% | 98.16% | 94.44% | 0 | 503 | PASS |
| 00000087 | 1,397 | 93.20% | 99.00% | 99.12% | 0 | 1063 | PASS |
| 00000094 | 3,218 | 85.02% | 91.27% | 74.00% | 2 | 1157 | PASS |

| Aggregate | Result |
|---|---|
| Successful responses | 10 / 10 |
| Eligible GPS observations | 19,491 |
| Weighted directed Top-1 before / after | 80.26% / 91.07% |
| Weighted undirected Top-1 after HMM | 94.84% |
| Route-switch continuity | 99.95% (3,919 connected / 2 disconnected) |
| Mean ordered sequence F1 | 80.67% |
| Mean server latency | 859.3 ms (contract test, not a physical LAN) |

Trip 00000015 shows point-level and route-level outcomes diverging: 82.99% Top-1 but only 21.07% sequence F1 with zero breaks. No further tuning is justified from these reserved-trip results.

> **Screenshots:** the Uvicorn log (`127.0.0.1`), Swagger UI and four UI captures (trips 00000015, 00000094, 00000049, 00000087) are **local integration** evidence. They must not be presented as a two-laptop demonstration. The current UI draws dense GPS with thick SVG marks that can hide the orange predicted route; use the layer toggles.

---

## 14. How to run

### Backend (Laptop 1, Ubuntu)

Run from the repository root with the model environment active:

```bash
PATHSNAP_MODEL=deployment/selected_model.joblib \
PATHSNAP_CONFIG=deployment/feature_config.json \
PATHSNAP_FEATURES=outputs/features/features_evaluator_test.parquet \
PATHSNAP_TRIPS=Data/processed/evaluator_test \
python -m uvicorn src.reconstruction.api_compatible:app \
  --host 0.0.0.0 --port 8000 --access-log
```

### Frontend (Laptop 2, Windows)

```powershell
# Stop any test/mock API first, then serve the current UI folder
py -m http.server 5500
# Open http://127.0.0.1:5500, enter http://<Ubuntu-LAN-IPv4>:8000 and click Connect

# Confirm connectivity first
Test-NetConnection <Ubuntu-IP> -Port 8000
```

- Use a shared, non-isolated network. Allow port 8000 only on the trusted private LAN; **do not expose the unauthenticated demo service publicly**.
- The Ubuntu terminal must show the **Windows client address** on incoming requests.
- Select any of the ten trips live and confirm prediction, route and metrics come back with no hardcoded answers.

---

## 15. Modelling assumptions

These started as initial assumptions and have since been checked against the data where noted.

| # | Assumption | Approach / status |
|---|---|---|
| 5.1 | GPS observations deviate from the true road | Generate several nearby candidates instead of assuming points lie on roads |
| 5.2 | The correct segment is among nearby candidates | STRtree radius search; **recall measured at 99.59%** |
| 5.3 | Consecutive observations are continuous movement, except large gaps | Use timestamps, displacement, speed, direction; segment on gaps > 30 s |
| 5.4 | Routes follow connected, direction-respecting road segments | Directed graph and transition penalties |
| 5.5 | Sampling intervals are irregular (7,286 training and 264 student gaps > 1 s; max 153,064 s) | Use actual timestamp differences, never a constant interval |
| 5.6 | `.route` length ≠ GPS count, so rows cannot be matched by position | Monotonic dynamic-programming alignment (proxy labels) |
| 5.7 | Lon/lat needs metric distances | Local UTM projection |
| 5.8 | `.nodes`/`.arcs` describe the available network; reverse arcs, restrictions and completeness need checking | Preserve arc directions; connectivity audited (multiple weak components in all 10 student graphs) |
| 5.9 | Reference routes must not influence unseen-trip predictions | Group-aware CV; student-test for development evaluation; evaluator data kept separate |
| 5.10 | Independent per-point prediction gives disconnected routes | Five supervised scorers + HMM/Viterbi; benefit verified experimentally (+4.60 pp Top-1, 258 → 5 disconnected switches) |

---

## 16. Repository structure

> The paths below combine the original project layout with the source paths cited in the technical report. Reconcile them with what is actually committed before submission.

```text
ML-Hackathon/
├── README.md
├── README_SUBMISSION.md
├── requirements.txt / requirements-pathsnap-final.txt
├── .gitignore
├── .env.example
│
├── configs/                 # data, models, bayes_opt, hmm configs
├── Data/
│   ├── raw/                 # train/, student_test/  (do not publish restricted data)
│   └── processed/           # incl. evaluator_test (private)
│
├── src/
│   ├── data/                # loader, preprocessing, projection, splits
│   ├── candidates/          # candidate generation (UTM + STRtree)
│   ├── features/            # 25-feature extraction (build_candidates.py)
│   ├── models/              # LR, weighted KNN, Random Forest, XGBoost, Deep MLP
│   ├── evaluation/          # metrics, plots
│   ├── tuning/              # Optuna TPE + grid search
│   ├── reconstruction/      # decoder.py (HMM/Viterbi), api_compatible.py (FastAPI app)
│   ├── viz/                 # Folium map
│   └── pipeline.py
│
├── deployment/
│   ├── selected_model.joblib
│   ├── selected_model_manifest.json
│   └── feature_config.json
│
├── artifacts/
│   ├── guideline_reports/   # validation/, all_models_cv/, student_test/, parameters/,
│   │                        # bayesian_comparison/, model_route_continuity/, final_submission_audit.md
│   ├── logistic_regression_bayesian/   # best_params.json, optuna_trials.csv, cv_oof_metrics.json
│   ├── logistic_regression_tuned/      # best_params.json, cv_oof_metrics.json, model_all_80.joblib
│   ├── reconstruction/student_test/    # comparison.json, per-trip predictions, expanded_route_arcs.csv,
│   │                                   # 00000000/road_network_map.html
│   └── api_checks/          # contract_summary.json
│
├── api/ and ui/             # API adapter / static browser UI (port 5500)
├── scripts/                 # render_folium_guidelines.py, recompute_development_diagnostics.py, run_all.sh
├── notebooks/               # 01_eda, 02_model_comparison, 03_hmm_analysis
├── results/                 # logs, tables, figures, maps, optuna
├── tests/                   # test_candidates, test_features, test_viterbi
└── report/                  # consolidated technical report PDF
```

**Repository checklist:** commit preprocessing, the feature generator, scripts for every model, grouped CV/evaluation, Bayesian search and logs, the HMM decoder, map code, API and UI source, selected weights/config and requirements. Do **not** publish credentials, evaluator reference files, reference-derived per-observation labels or datasets the organisers forbid. Confirm the repository opens publicly.

---

## 17. Limitations, integrity and remaining checks

**Limitations**

- Heading is unreliable when stationary or when sampling is irregular.
- Parallel roads and bidirectional arcs create competing candidates.
- Reference alignment is a **proxy** and can be ambiguous at junctions and revisits.
- The bounded graph connector can leave broken switches; sequence F1 is an ordered-arc diagnostic, not Fréchet distance or the official score.
- Family comparison mixes 64-trip fits with full-80 LR variants.
- Decoder parameters are defaults; no development-only sweep is evidenced.
- Timings come from the team's recorded environment, not a hardware-independent benchmark.
- UI thick GPS markers can hide the predicted path.

**Evaluation integrity.** The required protocol is: fit and optimise on the 80 training trips, select on the ten student-test trips, use evaluator data only after freezing. The selection manifest records `student_test_10` as the basis, the 25 features exclude reference labels, and the scoring code checks forbidden label columns. **However**, the development history includes evaluator-partition scoring and error inspection before the final selection manifest, so the project **cannot certify that the reserved partition stayed untouched**. The reserved results above are descriptive, not an independent pristine holdout. The team should review the chronology and disclose this to the evaluators.

**AI use and authorship.** AI assistance was used for code discussion, debugging/support-package preparation, report drafting, visualisation and the student-only diagnostic recomputation. All code, figures and conclusions must be independently understood and verified by the authors, and reconciled with the organiser's independent-work and AI-use policy.

### Compliance snapshot

| Checkpoint | Status |
|---|---|
| Dataset understanding / EDA | Documented |
| Five models and required comparison | Documented |
| Individual model analysis | Documented |
| Mandatory graphs | Documented |
| Bayesian optimisation | Documented |
| HMM / Viterbi | Documented |
| Folium | Documented (commit interactive source) |
| API/UI screenshots | Documented locally |
| **Physical two-laptop demo** | **OUTSTANDING** |
| **Evaluator-data integrity** | **REVIEW REQUIRED** |
| Independent work / AI disclosure | MANUAL REVIEW |
| One consolidated PDF | Prepared for review |
| Public GitHub submission | MANUAL REVIEW |

Do not describe the project as fully compliant until the physical demonstration and manual reviews are complete.

---

## 18. Defending the design: likely evaluator questions

**Q: Logistic Regression is a classifier. How can you use it for route prediction?**

> We formulated map matching as a binary candidate-classification problem rather than a continuous regression problem. For each GPS observation we generate nearby candidate road segments and compute 25 engineered features describing their geometric, directional, temporal and topological relationships.
>
> Logistic Regression estimates the probability that each candidate is the correct road segment. Those probabilities become observation scores in an HMM-style sequence model, and the Viterbi algorithm adds road connectivity and transition plausibility to reconstruct the most likely overall route.
>
> So Logistic Regression handles candidate classification, and HMM/Viterbi handles sequential route reconstruction.

**Q: Did the HMM actually help?**

| Metric (student-test, proxy reference) | Before HMM | After HMM |
|---|---:|---:|
| Conditional Top-1 accuracy | 85.92% | **90.53%** |
| Route continuity | 95.57% | **99.87%** |
| Disconnected arc switches | 258 | **5** |
| Mean ordered sequence F1 | 46.35% | **79.24%** |

**Q: Is LR the best model?** No. Random Forest has higher post-HMM Top-1 (91.36%) and XGBoost has better macro-F1 and log loss. LR was chosen for speed (0.064 s), simplicity, stable CV and competitive Top-1.

**Q: Why not just trust the nearest road?** Parallel roads and junctions make nearest-distance ambiguous; the classifier uses heading, motion and topology too, and the HMM adds connectivity.

**Q: Are your accuracy numbers the official score?** No. They are proxy-alignment metrics, labelled as such throughout.

---

## 19. Metric definitions

- **Binary metrics** use TN/FP/FN/TP at threshold 0.5 on usable candidate rows: accuracy = (TP+TN)/N; precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R). **Macro-F1** averages positive and negative class F1 equally. **Log loss** is the mean negative log-likelihood from clipped scores (lower is better).
- **Conditional Top-1** = eligible observations whose selected directed arc equals the reference-aligned arc ÷ eligible observations. Eligible means a usable aligned label with exactly one positive candidate. **Undirected Top-1** counts reversed direction as the same physical segment. **Weighted** Top-1 combines trips by eligible-point count.
- **Candidate miss rate** = 1 − fraction of pre-filter point groups whose candidates contain the reference-aligned arc (distinct from points excluded from Top-1).
- **Route-switch continuity** = connected distinct-arc switches ÷ (connected + disconnected), inside uninterrupted segments.
- **Ordered sequence F1** = 2m ÷ (len(predicted expanded arcs) + len(reference arcs)), with *m* matched ordered occurrences from `SequenceMatcher` (autojunk disabled). It is a diagnostic, not an official geospatial score. Mean sequence F1 is unweighted across trips.
- **Unique reference-arc coverage** = |predicted ∩ reference unique arcs| ÷ |reference unique arcs| (ignores order and repetition).
- **Latency:** the API log is server response time; browser round-trip adds transport and parsing.

---

*Sources: PES University PathSnap guideline PDFs; Student folder.zip; PathSnap_Report_Results.zip; team terminal history and screenshots, 18–19 November 2026; student-only paired route diagnostic recomputation.*
