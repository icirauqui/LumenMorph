"""Freeze a regular source-pixel cohort and anchor schedule from public metadata.
Run before inference. The output JSON and pixels.npy define its requested input
contract; fill every requested frame in the prediction, including failures.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from backend.src.benchmark_reader import PublicDataset
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--anchors',required=True,type=int,nargs='+');p.add_argument('--stream',default='moving');a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
public=PublicDataset(a.package);T=len(public.frames(a.stream));H,W=public.intrinsics.height,public.intrinsics.width
if any(x<0 or x>=T for x in a.anchors):raise ValueError('Anchor frame outside public sequence')
rr,cc=np.meshgrid(np.linspace(0,H-1,16,dtype=np.int64),np.linspace(0,W-1,16,dtype=np.int64),indexing='ij');pixels=np.column_stack((rr.ravel(),cc.ravel())).astype('<i8')
protocol=dict(schema='material_tracks_alignment_v1',component_id='single_full_sequence',reference_frame=0,frame_ids=list(range(T)),anchor_frame_ids=a.anchors,minimum_second_singular_ratio=1e-6,reference_pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
a.output.mkdir(parents=True);np.save(a.output/'pixels.npy',pixels);(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
