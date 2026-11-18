"""PathSnap candidate scorer + road-network HMM/Viterbi decoder.

Prediction uses candidate features and directed road geometry ONLY. Reference
routes and label columns are read only by `evaluate_predictions` afterwards.

This is a first-stage, segment-state HMM with directed shortest-path
transitions; fitted decoder weights should be selected using training-only CV.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import heapq
import math
import numpy as np
import pandas as pd
from pyproj import Transformer


@dataclass(frozen=True)
class Config:
    emission_weight: float = 1.0
    distance_beta_m: float = 30.0
    disconnected_penalty: float = 12.0
    uturn_penalty: float = 4.0
    gap_seconds: float = 30.0
    max_network_distance_m: float = 500.0
    min_network_distance_m: float = 100.0
    backslide_tolerance_m: float = 7.0


class RoadNetwork:
    def __init__(self, folder: Path):
        folder = Path(folder)
        t = folder.name
        self.trip_id = t
        nodes = np.loadtxt(folder / f'{t}.nodes', dtype=np.float64, ndmin=2)
        arcs = np.loadtxt(folder / f'{t}.arcs', dtype=np.int64, ndmin=2)
        if nodes.ndim != 2 or nodes.shape[1] != 2 or arcs.ndim != 2 or arcs.shape[1] != 2:
            raise ValueError(f'{t}: expected 2-column nodes and arcs')
        if arcs.min() < 0 or arcs.max() >= len(nodes):
            raise ValueError(f'{t}: invalid node references')
        lon = float(np.nanmean(nodes[:, 0]))
        lat = float(np.nanmean(nodes[:, 1]))
        zone = int(np.clip(np.floor((lon + 180) / 6) + 1, 1, 60))
        epsg = (32600 if lat >= 0 else 32700) + zone
        tr = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
        x, y = tr.transform(nodes[:, 0], nodes[:, 1])
        xy = np.column_stack((x, y))
        self.node_coords = nodes
        self.arcs = arcs
        self.lengths = np.linalg.norm(xy[arcs[:, 1]] - xy[arcs[:, 0]], axis=1)
        self.adj = [[] for _ in range(len(nodes))]
        for arc_id, ((start, end), length) in enumerate(zip(arcs, self.lengths)):
            if np.isfinite(length) and length > 0:
                self.adj[int(start)].append((int(end), float(length), arc_id))
        self.undirected_key = [tuple(sorted(map(int, e))) for e in arcs]

    def shortest(self, source, targets, limit, with_paths=False):
        """One bounded directed Dijkstra for all current targets, stops early."""
        source = int(source)
        target_set = set(map(int, targets))
        remaining = target_set - {source}
        distances = {source: 0.0}
        predecessors = {}
        found = {source: 0.0} if source in target_set else {}
        if not remaining:
            return found, predecessors
        heap = [(0.0, source)]
        while heap and remaining:
            d, node = heapq.heappop(heap)
            if d != distances.get(node) or d > limit:
                continue
            if node in remaining:
                found[node] = d
                remaining.remove(node)
                if not remaining:
                    break
            for nxt, w, aid in self.adj[node]:
                new = d + w
                if new <= limit and new < distances.get(nxt, math.inf):
                    distances[nxt] = new
                    if with_paths:
                        predecessors[nxt] = (node, aid)
                    heapq.heappush(heap, (new, nxt))
        return found, predecessors

    def connector(self, from_arc, to_arc, limit=3000.0):
        source = int(self.arcs[from_arc, 1]); target = int(self.arcs[to_arc, 0])
        if source == target:
            return []
        found, previous = self.shortest(source, [target], limit, with_paths=True)
        if target not in found:
            return None
        steps = []
        node = target
        while node != source:
            old, arc_id = previous[node]
            steps.append(int(arc_id))
            node = old
        return steps[::-1]


def score_candidates(df: pd.DataFrame, package: dict) -> pd.DataFrame:
    cols = package['feature_cols']
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f'Missing input features: {missing}')
    prohibited = {'label', 'label_usable', 'label_arc_id', 'label_route_pos',
                  'label_dist_m', 'point_has_positive'}
    if prohibited.intersection(cols):
        raise ValueError('Reference-derived labels are forbidden model inputs')
    X = (df[cols].apply(pd.to_numeric, errors='coerce')
         .replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32))
    scored = df[['trip_id', 'point_idx', 'arc_id', 'frac_along',
                 'gps_step_m', 'dt_s']].copy()
    scored['probability'] = package['model'].predict_proba(X)[:, 1].astype(np.float32)
    return scored


def decode_trip(candidates: pd.DataFrame, network: RoadNetwork,
                config: Config = Config()) -> tuple[pd.DataFrame, dict]:
    """Viterbi selection with directed road-network transition costs.

    A large GPS timestamp gap starts an independent decoding segment. At
    least one candidate per GPS point is required (generator supplies fallback).
    """
    needed = {'trip_id', 'point_idx', 'arc_id', 'probability',
              'frac_along', 'gps_step_m', 'dt_s'}
    missing = needed - set(candidates.columns)
    if missing:
        raise ValueError(f'Missing candidate fields: {sorted(missing)}')
    if candidates.empty:
        raise ValueError('No road candidates available')
    if candidates['trip_id'].nunique() != 1:
        raise ValueError('Decode one trip at a time')
    data = candidates.sort_values(['point_idx','arc_id'],kind='stable').reset_index(drop=True)
    if data.duplicated(['point_idx','arc_id']).any():
        raise ValueError('Duplicate point/arc candidate pairs')
    groups = [(int(point), group) for point,group in data.groupby('point_idx', sort=True)]
    points = [point for point,_ in groups]
    states = data['arc_id'].astype(int)
    if states.min()<0 or states.max()>=len(network.arcs):
        raise ValueError('Candidate road arc outside road network')
    return _decode_segments(data, network, config, points, groups)


def _decode_segments(candidates, network, config, points, groups):
    """Exact segment-aware Viterbi implementation (no segment boundary leakage)."""
    trip = str(candidates['trip_id'].iloc[0]).zfill(8)
    state_sets = [g['arc_id'].to_numpy(dtype=np.int64) for _, g in groups]
    frac_sets = [g['frac_along'].to_numpy(dtype=float) for _, g in groups]
    prob_sets = [np.clip(g['probability'].to_numpy(dtype=float), 1e-8, 1-1e-8) for _, g in groups]
    results = []
    boundaries = [0]
    for t in range(1, len(groups)):
        gps = float(groups[t][1]['gps_step_m'].iloc[0]); dt = float(groups[t][1]['dt_s'].iloc[0])
        if (points[t] != points[t-1]+1 or not np.isfinite(dt) or dt<=0
                or dt>config.gap_seconds or not np.isfinite(gps)):
            boundaries.append(t)
    boundaries.append(len(groups))
    for start, finish in zip(boundaries[:-1], boundaries[1:]):
        dp = config.emission_weight * np.log(prob_sets[start]); backs = [None]
        for t in range(start+1, finish):
            old, new = state_sets[t-1], state_sets[t]
            pf, nf = frac_sets[t-1], frac_sets[t]
            gps = max(0., float(groups[t][1]['gps_step_m'].iloc[0]))
            limit = min(config.max_network_distance_m,
                        max(config.min_network_distance_m, 3*gps+50))
            targets = np.unique(network.arcs[new, 0])
            found_cache = {}
            for src in np.unique(network.arcs[old, 1]):
                found_cache[int(src)] = network.shortest(src, targets, limit)[0]
            next_dp = np.full(len(new), -np.inf)
            back = np.zeros(len(new), dtype=np.int32)
            for j, aid in enumerate(new):
                start_node = int(network.arcs[aid, 0]); best = -np.inf; bi = 0
                for i, old_aid in enumerate(old):
                    if aid == old_aid and nf[j] >= pf[i]:
                        travel = (nf[j]-pf[i])*network.lengths[aid]; penalty = 0.
                    elif (aid == old_aid and
                          (pf[i]-nf[j])*network.lengths[aid] <= config.backslide_tolerance_m):
                        travel = (pf[i]-nf[j])*network.lengths[aid]; penalty = 1.
                    else:
                        middle = found_cache[int(network.arcs[old_aid, 1])].get(start_node)
                        if middle is None:
                            travel = gps; penalty = config.disconnected_penalty
                        else:
                            travel = ((1-pf[i])*network.lengths[old_aid] + middle
                                      + nf[j]*network.lengths[aid]); penalty = 0.
                        if (network.arcs[old_aid, 0] == network.arcs[aid, 1] and
                            network.arcs[old_aid, 1] == network.arcs[aid, 0]):
                            penalty += config.uturn_penalty
                    value = dp[i] - abs(travel-gps)/config.distance_beta_m - penalty
                    if value > best:
                        best = value; bi = i
                next_dp[j] = best + config.emission_weight*np.log(prob_sets[t][j])
                back[j] = bi
            next_dp -= max(next_dp)
            dp = next_dp; backs.append(back)
        states = np.zeros(finish-start, dtype=np.int32)
        s = int(np.argmax(dp))
        for local in range(finish-start-1, -1, -1):
            states[local] = s
            if local:
                s = int(backs[local][s])
        for local, choice in enumerate(states):
            g = groups[start+local][1].iloc[int(choice)]
            results.append({'trip_id':trip, 'point_idx':points[start+local],
                            'arc_id':int(g['arc_id']), 'probability':float(g['probability']),
                            'segment_id':0})
    # Assign monotonically increasing segment IDs from known boundaries.
    segment_ids = np.zeros(len(results), dtype=np.int32)
    for seg_idx, (start, finish) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segment_ids[start:finish] = seg_idx
    for i, value in enumerate(results):
        value['segment_id'] = int(segment_ids[i])
    selected = pd.DataFrame(results)
    baseline = candidates.loc[
        candidates.groupby('point_idx', sort=True)['probability'].idxmax(),
        ['point_idx', 'arc_id']
    ].set_index('point_idx')['arc_id']
    selected['baseline_arc_id'] = selected['point_idx'].map(baseline).astype(int)
    selected['changed'] = selected['arc_id'] != selected['baseline_arc_id']
    arc_path = []
    broken = 0; connected = 0; same = 0
    for seg, rows in selected.groupby('segment_id', sort=True):
        old = None
        for aid in rows['arc_id'].astype(int):
            if old is None:
                arc_path.append(int(aid))
            elif aid == old:
                same += 1
            else:
                path = network.connector(old, aid, limit=max(config.max_network_distance_m*2, 1000))
                if path is None:
                    broken += 1
                    arc_path.append(int(aid))
                else:
                    connected += 1
                    arc_path.extend(path)
                    arc_path.append(int(aid))
            old = int(aid)
    meta = {'trip_id':trip, 'gps_points':len(results),
            'segments':len(boundaries)-1, 'gap_resets':len(boundaries)-2,
            'changed_vs_classifier':int(selected['changed'].sum()),
            'changed_pct':float(selected['changed'].mean()),
            'connected_arc_switches':connected,
            'disconnected_arc_switches':broken,
            'same_arc_steps':same, 'expanded_route_arc_ids':arc_path,
            'expanded_route_arcs':len(arc_path)}
    return selected, meta


def evaluation_metrics(features: pd.DataFrame, selected: pd.DataFrame,
                       network: RoadNetwork) -> dict:
    """Metric ONLY: reference proxy labels never affect Viterbi inference."""
    df = features
    if not {'label_usable','label'}.issubset(df.columns):
        return {'available':False}
    df = df.loc[df['label_usable'].eq(True)].copy()
    if df.empty:
        return {'available':False}
    groups = df.groupby('point_idx')['label'].sum()
    good_points = groups.index[groups.eq(1)]
    truth = df.loc[df['point_idx'].isin(good_points) & df['label'].eq(1),
                   ['point_idx','arc_id']].rename(columns={'arc_id':'reference_arc_id'})
    if len(truth) != len(good_points):
        raise ValueError('Invalid point reference labels')
    pair = selected.merge(truth, on='point_idx', how='inner', validate='one_to_one')
    if pair.empty:
        return {'available':False}
    ref = pair['reference_arc_id'].astype(int).to_numpy()
    pred = pair['arc_id'].astype(int).to_numpy()
    base = pair['baseline_arc_id'].astype(int).to_numpy()
    geo = network.undirected_key
    def metric(a):
        return {'directed':float(np.mean(a == ref)),
                'undirected':float(np.mean([geo[x] == geo[y] for x,y in zip(a,ref)]))}
    return {'available':True, 'eligible_points':len(pair),
            'eligible_fraction':len(pair)/len(selected),
            'baseline':metric(base), 'hmm':metric(pred),
            'note':'Proxy reference alignment, not end-to-end route ground truth. '
                   'Undirected counts opposite directions as the same physical segment.'}


def geometry(network: RoadNetwork, selected: pd.DataFrame, meta: dict,
             track: np.ndarray, max_gps_points: int = 4000) -> dict:
    """GeoJSON-compatible map geometry, reference route NEVER included."""
    lines = []
    current = []
    last_end = None
    for aid in meta['expanded_route_arc_ids']:
        start, end = map(int, network.arcs[aid])
        p1, p2 = network.node_coords[[start,end]].tolist()
        if last_end is None or last_end != start:
            if len(current) > 1:
                lines.append(current)
            current = [p1, p2]
        else:
            current.append(p2)
        last_end = end
    if len(current) > 1:
        lines.append(current)
    coords = np.asarray(track, dtype=float)
    if len(coords) > max_gps_points:
        ids = np.unique(np.linspace(0, len(coords)-1, max_gps_points).astype(int))
        coords = coords[ids]
    return {
        'type':'FeatureCollection',
        'features':[
            {'type':'Feature','properties':{'layer':'raw_gps'},
             'geometry':{'type':'LineString','coordinates':coords[:, :2].tolist()}},
            {'type':'Feature','properties':{'layer':'predicted_route'},
             'geometry':{'type':'MultiLineString','coordinates':lines}},
        ]
    }


def load_gps(folder: Path) -> np.ndarray:
    t=folder.name
    processed=folder / f'{t}_track_processed.csv'
    if processed.exists():
        df=pd.read_csv(processed,usecols=['longitude','latitude'])
        return df[['longitude','latitude']].to_numpy(float)
    return np.loadtxt(folder / f'{t}.track',ndmin=2)[:, :2]


def sequence_metrics(network: RoadNetwork, meta: dict, folder: Path) -> dict:
    """Server-side, post-prediction route sequence comparison. Never used by decoder."""
    from difflib import SequenceMatcher
    path=Path(folder) / f'{network.trip_id}.route'
    if not path.exists():
        return {'available':False}
    reference=np.loadtxt(path,dtype=np.int64,ndmin=1).tolist()
    prediction=list(map(int,meta['expanded_route_arc_ids']))
    if not reference or not prediction:
        return {'available':False}
    ref_set=set(reference); pred_set=set(prediction)
    overlap=len(ref_set & pred_set)
    # LCS via matching blocks is a diagnostic, not a fully equivalent route-distance metric.
    ordered_match=SequenceMatcher(a=reference,b=prediction,autojunk=False)
    n_common=sum(block.size for block in ordered_match.get_matching_blocks())
    precision=n_common/len(prediction)
    recall=n_common/len(reference)
    score=2*precision*recall/(precision+recall) if precision+recall else 0.
    return {'available':True,'reference_route_arcs':len(reference),
            'predicted_expanded_arcs':len(prediction),
            'reference_unique_arc_coverage':overlap/len(ref_set),
            'prediction_unique_arc_precision':overlap/len(pred_set),
            'ordered_sequence_match_f1':score,
            'note':'Post-prediction directed arc-sequence similarity (matching blocks); '
                   'not geometric Fréchet distance or official route reconstruction accuracy.'}
