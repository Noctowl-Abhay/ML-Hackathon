"""Small deterministic synthetic route test; no reference label is passed to decoder."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from src.reconstruction.decoder import RoadNetwork, decode_trip, Config, geometry


def test_connected_route():
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)/'00000001';folder.mkdir()
        np.savetxt(folder/'00000001.nodes',[[77,12],[77.0001,12],[77.0002,12]],fmt='%.8f')
        np.savetxt(folder/'00000001.arcs',[[0,1],[1,2],[1,0],[2,1]],fmt='%d')
        frame=pd.DataFrame([
            # local point 0: arc 0 best
            dict(trip_id='00000001',point_idx=0,arc_id=0,frac_along=.7,gps_step_m=0.,dt_s=1.,probability=.92),
            dict(trip_id='00000001',point_idx=0,arc_id=2,frac_along=.3,gps_step_m=0.,dt_s=1.,probability=.4),
            # local point 1: arc 1 best
            dict(trip_id='00000001',point_idx=1,arc_id=1,frac_along=.3,gps_step_m=6.,dt_s=1.,probability=.88),
            dict(trip_id='00000001',point_idx=1,arc_id=3,frac_along=.7,gps_step_m=6.,dt_s=1.,probability=.3),
            # local point 2: stays on arc 1
            dict(trip_id='00000001',point_idx=2,arc_id=1,frac_along=.8,gps_step_m=5.,dt_s=1.,probability=.94),
        ])
        road=RoadNetwork(folder)
        out, meta=decode_trip(frame,road,Config())
        assert out['arc_id'].tolist()==[0,1,1],out
        assert meta['disconnected_arc_switches']==0, meta
        assert meta['expanded_route_arc_ids']==[0,1],meta
        assert geometry(road,out,meta,np.array([[77,12],[77.0001,12],[77.0002,12]]))['features'][1]['geometry']['type']=='MultiLineString'


def test_gap_reset():
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)/'00000001';folder.mkdir()
        np.savetxt(folder/'00000001.nodes',[[77,12],[77.0001,12]],fmt='%.8f')
        np.savetxt(folder/'00000001.arcs',[[0,1],[1,0]],fmt='%d')
        df=pd.DataFrame([
            dict(trip_id='00000001',point_idx=0,arc_id=0,frac_along=.4,gps_step_m=0.,dt_s=1.,probability=.9),
            dict(trip_id='00000001',point_idx=1,arc_id=1,frac_along=.7,gps_step_m=0.,dt_s=120.,probability=.9),
        ])
        pred,stats=decode_trip(df,RoadNetwork(folder),Config())
        assert stats['gap_resets']==1
        assert pred['segment_id'].tolist()==[0,1]
