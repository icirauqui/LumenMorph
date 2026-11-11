"""Dense scoring in the appearance renderer's metre-valued world.

Projected flow scores source-valid, forward/in-image material trajectories;
it does not claim target visibility or baseline occlusion equivalence.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .appearance_reader import AppearanceEvaluation
from .benchmark_evaluate import digest,score_dense


def evaluate(family,predictions,output,task,frame,target=None,coordinate_frame=None):
    family,predictions,output=map(lambda p:Path(p).resolve(),(family,predictions,output))
    if family==output or family in output.parents:raise ValueError('Output must be outside immutable family')
    if output.exists():raise FileExistsError(output)
    if task not in ('depth','normals','displacement','projected_flow'):raise ValueError('Unsupported appearance task')
    if task in ('normals','displacement') and coordinate_frame!='appearance-world-meters':
        raise ValueError('Explicit appearance-world-meters frame required')
    if (task in ('displacement','projected_flow')) != (target is not None):raise ValueError('Target required only for temporal tasks')
    h=digest(predictions)
    with np.load(predictions,allow_pickle=False) as z:
        if not {'prediction'}<=set(z.files)<= {'prediction','mask'}:raise ValueError('Invalid prediction fields')
        pred=z['prediction'].copy();mask=z['mask'].copy() if 'mask' in z else None
    reader=AppearanceEvaluation(family);d=reader.dense(frame);eligible=d['valid'].copy();source_count=int(eligible.sum())
    if task=='depth':truth=d['depth_z']
    elif task=='normals':truth=d['normal_world']
    else:
        ids=d['triangle_id'][eligible];weights=d['material_barycentric'][eligible]
        if task=='displacement':
            truth=np.full(eligible.shape+(3,),np.nan)
            truth[eligible]=reader.material_points(target,ids,weights)-reader.material_points(frame,ids,weights)
        else:
            projected=reader.project_material(target,ids,weights);row,col=np.nonzero(eligible)
            truth=np.full(eligible.shape+(2,),np.nan);truth[eligible]=projected['uv']-np.column_stack((col+.5,row+.5))
            eligible[row,col]=projected['forward']&projected['inside_image']
    score=score_dense(pred,truth,eligible,'flow' if task=='projected_flow' else task,mask)
    if task in ('depth','displacement'):score['error_units']='meters'
    if digest(predictions)!=h:raise ValueError('Prediction mutated during scoring')
    inputs=[family/'public/frames.json',family/'public/intrinsics.json',family/'evaluation/states.npz',family/'evaluation/geometry.npz',family/'evaluation/dense'/reader._frame(frame)['image'].replace('.png','.npz')]
    out=dict(schema='appearance_prediction_score_v1',task=task,source_frame=frame,target_frame=target,
             coordinate_frame=coordinate_frame,source_valid_pixels=source_count,primary=score,
             visibility='not classified',population='source valid; projected flow additionally target forward and in-image',
             prediction_sha256=h,truth_inputs_sha256={str(p.relative_to(family)):digest(p) for p in inputs},evaluator_sha256=digest(__file__))
    output.parent.mkdir(parents=True,exist_ok=True);
    with output.open('x') as handle:handle.write(json.dumps(out,indent=2,allow_nan=False)+'\n')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['family','predictions','output','task','frame']:p.add_argument('--'+name,required=True,type=int if name=='frame' else str)
    p.add_argument('--target',type=int);p.add_argument('--coordinate-frame');a=p.parse_args();evaluate(**vars(a))

if __name__=='__main__':main()
