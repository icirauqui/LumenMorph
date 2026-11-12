import ctypes as c
from pathlib import Path
import numpy as np
class Rays:
 def __init__(self,vertices,triangles):
  self.handle=None
  vertices=np.asarray(vertices);triangles=np.asarray(triangles)
  if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():raise ValueError('Finite N-by-3 vertices required')
  if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.dtype.kind not in 'iu' or triangles.min()<0 or triangles.max()>=len(vertices):raise ValueError('Nonempty valid integer triangle topology required')
  self.lib=c.CDLL(str(Path(__file__).with_name('native_rays.so')))
  f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
  self.lib.make_scene.argtypes=[f64,i64,c.c_int64];self.lib.make_scene.restype=c.c_void_p
  self.lib.trace.argtypes=[c.c_void_p,f64,f64,c.c_int64,i64,f64,f64];self.lib.free_scene.argtypes=[c.c_void_p];self.lib.free_scene.restype=None;self.lib.trace.restype=None
  self.handle=self.lib.make_scene(np.ascontiguousarray(vertices,dtype=np.float64),np.ascontiguousarray(triangles,dtype=np.int64),len(triangles))
 def __call__(self,origin,directions):
  origin=np.asarray(origin);directions=np.asarray(directions)
  if not self.handle or origin.shape!=(3,) or directions.ndim!=2 or directions.shape[1]!=3 or not np.isfinite(origin).all() or not np.isfinite(directions).all():raise ValueError('Open scene, finite origin and N-by-3 directions required')
  n=len(directions);ids=np.empty(n,np.int64);z=np.empty(n);b=np.empty((n,3))
  self.lib.trace(self.handle,np.ascontiguousarray(origin,dtype=np.float64),np.ascontiguousarray(directions,dtype=np.float64),n,ids,z,b);return ids,z,b
 def close(self):
  if self.handle:self.lib.free_scene(self.handle);self.handle=None
 def __del__(self):self.close()
