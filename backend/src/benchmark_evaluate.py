"""Evaluate supplied dense predictions against an existing tubular package.

No estimator is run and no truth-based alignment is performed. Prediction masks
describe missing output, never select the benchmark's eligible population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .benchmark_reader import EvaluationDataset

TASKS = ('depth', 'flow', 'normals', 'displacement')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def score_dense(prediction, truth, eligible, task, availability=None):
    """Full-population scores are null unless every eligible prediction is valid.

    Normals are compared by direction in degrees; zero vectors are invalid.
    Scalar depth uses absolute error, vector flow/displacement endpoint error.
    Conditional errors remain explicitly labelled when predictions are missing.
    """
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}')
    prediction, truth, eligible = map(np.asarray, (prediction, truth, eligible))
    expected = eligible.shape if task == 'depth' else eligible.shape + ((2,) if task == 'flow' else (3,))
    if eligible.ndim != 2 or eligible.dtype != np.bool_ or prediction.shape != expected or truth.shape != expected:
        raise ValueError('Expected HxW boolean eligibility and matching native-resolution prediction/truth shapes')
    if not np.issubdtype(prediction.dtype, np.number) or np.iscomplexobj(prediction):
        raise ValueError('Predictions must be real numeric values')
    if availability is None:
        availability = np.ones(eligible.shape, bool)
    availability = np.asarray(availability)
    if availability.shape != eligible.shape or availability.dtype != np.bool_:
        raise ValueError('Optional prediction mask must be boolean HxW')
    truth_finite = np.isfinite(truth) if task == 'depth' else np.isfinite(truth).all(axis=-1)
    if not truth_finite[eligible].all():
        raise ValueError('Nonfinite truth inside the fixed eligible population')
    finite = np.isfinite(prediction) if task == 'depth' else np.isfinite(prediction).all(axis=-1)
    semantic = finite.copy()
    if task == 'depth':
        if (truth[eligible] <= 0).any():
            raise ValueError('Depth truth must be positive camera-z')
        semantic &= prediction > 0
    if task == 'normals':
        if (np.max(np.abs(truth[eligible].astype(float)), axis=-1) == 0).any():
            raise ValueError('Zero normal in eligible truth')
        semantic &= np.max(np.abs(prediction.astype(float)), axis=-1) > 0
    supported = eligible & availability & semantic
    n, m = int(eligible.sum()), int(supported.sum())
    out = dict(eligible_pixels=n, available_pixels=int((eligible & availability).sum()),
               finite_pixels=int((eligible & availability & finite).sum()),
               valid_prediction_pixels=m, missing_or_invalid_pixels=n-m,
               invalid_finite_pixels=int((eligible & availability & finite & ~semantic).sum()),
               prediction_coverage=float(m/n) if n else None,
               complete_prediction_support=bool(n and m == n),
               error_units={'depth':'scene_length','flow':'pixels','normals':'degrees','displacement':'scene_length'}[task])
    if m:
        p, t = prediction[supported].astype(float), truth[supported].astype(float)
        if task == 'normals':
            # Scaling first avoids overflow for finite, non-unit user normals.
            p /= np.max(np.abs(p),axis=-1,keepdims=True)
            t /= np.max(np.abs(t),axis=-1,keepdims=True)
            p /= np.linalg.norm(p, axis=-1, keepdims=True)
            t /= np.linalg.norm(t, axis=-1, keepdims=True)
            error = np.degrees(np.arccos(np.clip(np.sum(p*t, axis=-1), -1., 1.)))
        else:
            with np.errstate(over='ignore',invalid='ignore'):
                error = np.abs(p-t) if task == 'depth' else np.linalg.norm(p-t, axis=-1)
        if not np.isfinite(error).all():
            raise ValueError('Prediction errors exceed supported floating-point range')
        conditional = dict(mean=float(error.mean()), median=float(np.median(error)), p95=float(np.percentile(error,95)))
        if not all(np.isfinite(v) for v in conditional.values()):
            raise ValueError('Error summaries exceed supported floating-point range')
    else:
        conditional = dict(mean=None, median=None, p95=None)
    out['conditional_error'] = conditional
    out['full_population_error'] = conditional.copy() if out['complete_prediction_support'] else dict(mean=None,median=None,p95=None)
    return out


def evaluate(package, predictions, output, task, frame, stream='moving', target=None, coordinate_frame=None):
    package, predictions, output = map(lambda p: Path(p).resolve(), (package,predictions,output))
    suite = package.parent
    if output == suite or suite in output.parents:
        raise ValueError('Evaluation output must be outside the immutable suite')
    if task not in TASKS:
        raise ValueError('Unknown task')
    if task in ('normals','displacement') and coordinate_frame != 'benchmark-world':
        raise ValueError('Explicit --coordinate-frame benchmark-world is required; no truth alignment is performed')
    if task == 'flow' and target not in (None,frame+1):
        raise ValueError('Stored flow evaluates only target=source+1')
    if task == 'displacement' and target is None:
        raise ValueError('Displacement requires a stored target frame')
    if task in ('depth','normals') and target is not None:
        raise ValueError('Depth/normals have no target frame')
    prediction_hash = digest(predictions)
    # Load submitted output before constructing the privileged evaluator.
    with np.load(predictions, allow_pickle=False) as z:
        if set(z.files) - {'prediction','mask'} or 'prediction' not in z.files:
            raise ValueError('Prediction NPZ must contain prediction and optionally mask only')
        prediction = z['prediction'].copy()
        availability = z['mask'].copy() if 'mask' in z.files else None
    reader = EvaluationDataset(package)
    dense = reader.dense(frame,stream)
    eligible = dense['valid']
    inputs = [package/'DATASET.json',package/'public/intrinsics.json',
              reader.public_paths(stream)['frames'],reader.truth_path(stream)/'dense'/f'{frame:06d}.npz']
    strata = {}
    if task == 'depth':
        truth = dense['depth_z']
    elif task == 'normals':
        truth = dense['normal_world']
    elif task == 'flow':
        target = frame+1
        reader._frame(stream,target)
        path = reader.truth_path(stream)/'flow_forward'/f'{frame:06d}.npz'
        inputs.append(path)
        with np.load(path,allow_pickle=False) as z:
            flow = {key:z[key] for key in z.files}
        truth = flow['flow_uv']
        union = sum(flow[k].astype(np.uint8) for k in ('visible','occluded','out_of_frame','behind_camera','unresolved'))
        if np.any(union>1) or not np.array_equal(union==1,eligible):
            raise ValueError('Flow categories disagree with source geometry validity')
        strata = {s:eligible & flow['flow_valid'] & flow[s] for s in ('occluded','out_of_frame')}
        eligible = eligible & flow['flow_valid'] & flow['visible']
    else:
        reader._frame(stream,target)
        ids,weights = dense['triangle_id'][eligible],dense['material_barycentric'][eligible]
        truth = np.full(eligible.shape+(3,),np.nan)
        truth[eligible] = reader.material_displacement(ids,weights,target,stream,from_frame=frame)
        inputs += [reader.truth_path(stream)/'states.npz',package/'evaluation/family/geometry.npz']
    scores = score_dense(prediction,truth,eligible,task,availability)
    extra = {s:score_dense(prediction,truth,mask,task,availability) for s,mask in strata.items()}
    if digest(predictions) != prediction_hash:
        raise ValueError('Prediction file changed during evaluation')
    result = dict(schema='tubular_dense_prediction_evaluation_v1',task=task,package=str(package),stream=stream,
                  source_frame=frame,target_frame=target,coordinate_frame=coordinate_frame,
                  source_valid_pixels=int(dense['valid'].sum()),image_pixels=int(dense['valid'].size),
                  primary=scores,secondary_strata=extra,prediction_sha256=prediction_hash,
                  evaluator_sha256=digest(__file__),truth_inputs_sha256={str(p):digest(p) for p in inputs},
                  reader_sha256=digest(Path(__file__).with_name('benchmark_reader.py')),
                  population_rule='fixed truth validity; prediction mask only marks unavailable output; no truth alignment or method-specific eligibility',
                  metrics_scope='depth absolute; normal angular; flow/displacement endpoint error; no division by true displacement')
    if task == 'flow':
        result['visibility_counts']={s:int(flow[s].sum()) for s in ('flow_valid','visible','occluded','out_of_frame','behind_camera','unresolved')}
    if (suite/'SUITE_CHECKSUMS.json').is_file():
        result['suite_seal_sha256']=digest(suite/'SUITE_CHECKSUMS.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True,type=Path)
    p.add_argument('--predictions',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--task',required=True,choices=TASKS)
    p.add_argument('--frame',required=True,type=int)
    p.add_argument('--stream',choices=('reference','moving','stationary'),default='moving')
    p.add_argument('--target',type=int)
    p.add_argument('--coordinate-frame',choices=('benchmark-world',))
    args=p.parse_args()
    result=evaluate(args.package,args.predictions,args.output,args.task,args.frame,args.stream,args.target,args.coordinate_frame)
    print(json.dumps(result['primary'],indent=2))


if __name__=='__main__':
    main()
