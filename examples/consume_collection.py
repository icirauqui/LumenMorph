"""Read RGB, freeze a zero-displacement prediction, query and evaluate material.

This is an interface example, not a reconstruction algorithm or its evaluation.
Use the collection root or the self-contained example's root.
"""
import argparse,json
from pathlib import Path
import numpy as np
from backend.src.collection_reader import Collection
from backend.src.benchmark_reader import EvaluationDataset
from backend.src.benchmark_evaluate import evaluate

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--stream',default='f07_fem_moving_tissue');p.add_argument('--frame',type=int,default=0);p.add_argument('--target',type=int,default=1);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True)
collection=Collection(a.root);record=collection.streams[a.stream];image=collection.image_rgb(a.stream,a.frame)
height,width=image.shape[:2];prediction=a.output/'zero_displacement.npz'
np.savez_compressed(prediction,prediction=np.zeros((height,width,3),dtype=np.float64))
# Prediction is saved before opening evaluation-only files.
package=a.root/record['path'];reader=EvaluationDataset(package);dense=reader.dense(a.frame,record['acquisition'])
row,col=np.argwhere(dense['valid'])[len(np.argwhere(dense['valid']))//2]
triangle=dense['triangle_id'][row,col];weights=dense['material_barycentric'][row,col]
trajectory=reader.material_points(triangle,weights,a.target,record['acquisition'])
result=evaluate(package,prediction,a.output/'score.json','displacement',a.frame,record['acquisition'],a.target,'benchmark-world')
(a.output/'example.json').write_text(json.dumps(dict(rgb_shape=list(image.shape),rgb_dtype=str(image.dtype),source_pixel=[int(row),int(col)],material_target=trajectory.tolist(),scope='zero-prediction interface example, not inference'),indent=2)+'\n')
print(json.dumps(result['primary'],indent=2))
