"""Hash-identified external material textures, sampled without lighting claims."""
from pathlib import Path
import hashlib
import cv2
import numpy as np


def load_texture(root, spec):
    path=Path(root)/spec['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
        raise ValueError('External texture hash mismatch')
    if spec['color_space']!='srgb' or spec['interpolation'] not in ('nearest','bilinear'):
        raise ValueError('Supported: sRGB input with nearest or linear-light bilinear sampling')
    if spec['seam']!='periodic' or spec['orientation'] not in (0,90,180,270):
        raise ValueError('Texture needs periodic wrapping and quarter-turn orientation')
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('Cannot decode external texture')
    image=np.rot90(image,spec['orientation']//90).copy()
    return image


def sample(texture_bgr, uv, spec):
    """UV is persistent unwrapped material position; outputs encoded uint8 BGR."""
    uv=np.asarray(uv,float)
    repeat=np.asarray(spec['repeat'],float);offset=np.asarray(spec.get('offset',[0,0]),float)
    if repeat.shape!=(2,) or not np.isfinite(repeat).all() or np.any(repeat<=0):
        raise ValueError('repeat must contain two positive finite values')
    uv=(uv*repeat+offset)%1
    h,w=texture_bgr.shape[:2]
    # Pixel-centered periodic texture, V increases opposite image row.
    x=uv[:,0]*w-.5;y=(1-uv[:,1])*h-.5
    if spec['interpolation']=='nearest':
        return texture_bgr[np.floor(y+.5).astype(int)%h,np.floor(x+.5).astype(int)%w]
    linear=texture_bgr.astype(float)/255
    linear=np.where(linear<=.04045,linear/12.92,((linear+.055)/1.055)**2.4)
    ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);a=(x-ix)[:,None];b=(y-iy)[:,None]
    col=(1-a)*(1-b)*linear[iy%h,ix%w]+a*(1-b)*linear[iy%h,(ix+1)%w]+(1-a)*b*linear[(iy+1)%h,ix%w]+a*b*linear[(iy+1)%h,(ix+1)%w]
    col=np.where(col<=.0031308,12.92*col,1.055*np.maximum(col,0)**(1/2.4)-.055)
    return np.rint(np.clip(col,0,1)*255).astype(np.uint8)


def image_from_material(buffers, triangles, uv_vertices, texture_bgr, spec):
    valid=buffers['valid'];uvw=uv_vertices[triangles[buffers['triangle_id'][valid]]].astype(float)
    u=uvw[:,:,0]%1;u += ((np.ptp(u,axis=1)>.5)[:,None] & (u<.5))
    uvw[:,:,0]=u
    uv=np.sum(buffers['material_barycentric'][valid,:,None]*uvw,axis=1)
    image=np.full(valid.shape+(3,),12,np.uint8)
    image[valid]=sample(texture_bgr,uv,spec)
    return image
