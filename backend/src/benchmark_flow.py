"""Fixed development-only optical-flow usage study; never writes into a suite.

Run ``python -m backend.src.benchmark_flow --help`` from the generator checkout.
The protocol intentionally reports method failures and support, not data selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

SOURCE_FRAMES = (0, 14, 22, 37, 52, 67, 82, 88)
FARNEBACK = dict(pyr_scale=.5, levels=3, winsize=15, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0)
STRATA = ('visible', 'occluded', 'out_of_frame')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def score(prediction, target, mask):
    """Report finite support explicitly; empty support has null accuracy."""
    if prediction.shape != target.shape or target.shape != mask.shape + (2,):
        raise ValueError('Flow or mask shape mismatch')
    if mask.dtype != np.bool_:
        raise ValueError('Mask must be boolean')
    eligible = mask & np.isfinite(target).all(axis=-1)
    finite = eligible & np.isfinite(prediction).all(axis=-1)
    n = int(eligible.sum())
    result = {'eligible': n, 'predicted': int(finite.sum()),
              'prediction_coverage': float(finite.sum() / n) if n else None,
              'mean_truth_magnitude': float(np.linalg.norm(target[eligible], axis=-1).mean()) if n else None}
    if not finite.any():
        result.update({key: None for key in ('mean_epe', 'median_epe', 'p95_epe',
                      'fraction_gt1px', 'fraction_gt3px')})
        return result
    errors = np.linalg.norm(prediction[finite].astype(float) - target[finite], axis=-1)
    result.update(mean_epe=float(errors.mean()), median_epe=float(np.median(errors)),
                  p95_epe=float(np.percentile(errors, 95)),
                  fraction_gt1px=float(np.mean(errors > 1)),
                  fraction_gt3px=float(np.mean(errors > 3)))
    return result


def run(root, output, protocol):
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise ValueError('Results must be outside the immutable suite')
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    jobs, inputs = [], {}
    for version in range(3, 12):
        package = root / f'v{version}'
        descriptor = json.loads((package / 'DATASET.json').read_text())
        if descriptor['split'] != 'development':
            raise ValueError('This protocol is restricted to development packages')
        inputs[f'v{version}/DATASET.json'] = digest(package / 'DATASET.json')
        for stream in ('moving', 'stationary'):
            for frame in SOURCE_FRAMES:
                rgb = package / 'public' / stream / 'rgb'
                truth = package / 'evaluation' / stream / 'flow_forward' / f'{frame:06d}.npz'
                paths = (rgb / f'{frame:06d}.png', rgb / f'{frame+1:06d}.png', truth)
                for path in paths:
                    inputs[str(path.relative_to(root))] = digest(path)
                jobs.append((version, descriptor['family'], descriptor['condition'], stream, frame, paths))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    getters = ('FinestScale', 'GradientDescentIterations', 'PatchSize', 'PatchStride',
               'UseMeanNormalization', 'UseSpatialPropagation', 'VariationalRefinementAlpha',
               'VariationalRefinementDelta', 'VariationalRefinementGamma',
               'VariationalRefinementIterations')
    frozen = {'schema': 'tubular_flow_study_v1', 'protocol_sha256': digest(protocol),
              'runner_sha256': digest(__file__), 'suite_seal_sha256': digest(root / 'SUITE_CHECKSUMS.json'),
              'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
              'opencv_build': cv2.getBuildInformation(), 'threads': 1, 'rng_seed': 0,
              'farneback': FARNEBACK, 'dis_preset': 'MEDIUM',
              'dis_parameters': {key: getattr(dis, f'get{key}')() for key in getters},
              'source_frames': SOURCE_FRAMES, 'pair_count': len(jobs), 'inputs': inputs}
    write_json(output / 'FREEZE.json', frozen)
    rows = []
    with (output / 'pairs.jsonl').open('x') as log:
        for version, family, condition, stream, frame, paths in jobs:
            a, b = (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths[:2])
            if a is None or b is None:
                raise FileNotFoundError(paths[:2])
            predictions, times, errors = {}, {}, {}
            for method in ('zero', 'farneback', 'dis_medium'):
                start = time.perf_counter()
                try:
                    if method == 'zero':
                        predictions[method] = np.zeros(a.shape + (2,), np.float32)
                    elif method == 'farneback':
                        predictions[method] = cv2.calcOpticalFlowFarneback(a, b, None, **FARNEBACK)
                    else:
                        # A new object for each pair prevents undocumented temporal state.
                        predictor = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                        predictions[method] = predictor.calc(a, b, None)
                except Exception as exc:
                    predictions[method] = np.full(a.shape + (2,), np.nan, np.float32)
                    errors[method] = f'{type(exc).__name__}: {exc}'
                times[method] = time.perf_counter() - start
            # Truth is opened only after all predictions have been produced.
            with np.load(paths[2], allow_pickle=False) as z:
                truth = {k: z[k] for k in z.files}
            counts = sum(truth[k].astype(np.uint8) for k in (*STRATA, 'behind_camera', 'unresolved'))
            if np.any(counts > 1) or not np.array_equal(
                    counts == 1, truth['flow_valid'] | truth['behind_camera']):
                raise ValueError('Truth visibility categories violate the source-mask contract')
            row = dict(version=version, family=family, condition=condition, stream=stream,
                       source_frame=frame, pixels=int(a.size),
                       support={k: int(truth[k].sum()) for k in ('flow_valid', *STRATA, 'behind_camera', 'unresolved')},
                       methods={}, errors=errors, runtime_seconds=times)
            row['source_valid'] = sum(row['support'][k] for k in (*STRATA, 'behind_camera', 'unresolved'))
            for method, prediction in predictions.items():
                row['methods'][method] = {s: score(prediction, truth['flow_uv'], truth[s] & truth['flow_valid'])
                                           for s in STRATA}
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            rows.append(row)
            if len(rows) % 16 == 0:
                print(f'{len(rows)}/{len(jobs)} pairs', flush=True)
    aggregate = []
    for condition in ('static', 'fem', 'nonfem'):
        for stream in ('moving', 'stationary'):
            for method in ('zero', 'farneback', 'dis_medium'):
                selected = [r for r in rows if r['condition'] == condition and r['stream'] == stream]
                means = []
                for family in sorted({r['family'] for r in selected}):
                    metrics = [r['methods'][method]['visible'] for r in selected if r['family'] == family]
                    valid = [m['mean_epe'] for m in metrics if m['mean_epe'] is not None]
                    complete = len(valid) == len(SOURCE_FRAMES) and all(m['prediction_coverage'] == 1 for m in metrics)
                    means.append({'family': family, 'expected_pairs': len(SOURCE_FRAMES),
                                  'scored_pairs': len(valid), 'complete_prediction_support': complete,
                                  'conditional_mean_epe': float(np.mean(valid)) if valid else None,
                                  'mean_epe': float(np.mean(valid)) if complete else None})
                vals = [m['mean_epe'] for m in means if m['mean_epe'] is not None]
                aggregate.append(dict(condition=condition, stream=stream, method=method,
                                      families=means, mean_pair_epe=float(np.mean(vals)) if len(vals) == len(means) else None,
                                      failed_pairs=sum(method in r['errors'] for r in selected)))
    write_json(output / 'RESULT.json', {'schema': 'tubular_flow_study_v1', 'pairs': len(rows),
               'expected_pairs': len(jobs), 'failed_calls': sum(len(r['errors']) for r in rows),
               'aggregation': 'equal family means of per-pair visible mean EPE; three development families',
               'aggregate': aggregate, 'freeze_sha256': digest(output / 'FREEZE.json'),
               'pairs_sha256': digest(output / 'pairs.jsonl')})
    # Detect data changes during the run without retaining duplicate inputs.
    changed = [p for p, expected in inputs.items() if digest(root / p) != expected]
    write_json(output / 'READBACK.json', {'unchanged': not changed, 'changed': changed,
               'input_count': len(inputs), 'result_sha256': digest(output / 'RESULT.json')})
    if changed:
        raise RuntimeError('Inputs changed during scoring')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    args = p.parse_args()
    run(args.root, args.output, args.protocol)


if __name__ == '__main__':
    main()
