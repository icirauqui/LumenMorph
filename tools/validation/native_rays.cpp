// Independent double-precision world-space ray/triangle BVH. No renderer code.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>
struct V { double x[3]; double& operator[](int i){return x[i];} double operator[](int i)const{return x[i];} };
V sub(V a,V b){return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}};}
V cross(V a,V b){return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
struct Face {V p,e1,e2,lo,hi,center; double scale; int64_t id;};
struct Node {V lo,hi; int begin,end,left=-1,right=-1;};
struct Scene {
 std::vector<Face> f;std::vector<int> order;std::vector<Node> nodes;
 int build(int a,int b){
  Node n;n.begin=a;n.end=b;
  for(int j=0;j<3;j++){n.lo[j]=INFINITY;n.hi[j]=-INFINITY;}
  for(int i=a;i<b;i++)for(int j=0;j<3;j++){n.lo[j]=std::min(n.lo[j],f[order[i]].lo[j]);n.hi[j]=std::max(n.hi[j],f[order[i]].hi[j]);}
  int id=nodes.size();nodes.push_back(n);
  if(b-a>8){int axis=0;for(int j=1;j<3;j++)if(n.hi[j]-n.lo[j]>n.hi[axis]-n.lo[axis])axis=j;
   int m=(a+b)/2;std::nth_element(order.begin()+a,order.begin()+m,order.begin()+b,[&](int i,int k){return f[i].center[axis]<f[k].center[axis];});
   int l=build(a,m),r=build(m,b);nodes[id].left=l;nodes[id].right=r;
  }return id;
 }
 bool box(const Node& n,V o,V d,double best){
  double lo=1e-5,hi=best;
  for(int j=0;j<3;j++){
   if(d[j]==0){if(o[j]<n.lo[j]||o[j]>n.hi[j])return false;continue;}
   double a=(n.lo[j]-o[j])/d[j],b=(n.hi[j]-o[j])/d[j];if(a>b)std::swap(a,b);lo=std::max(lo,a);hi=std::min(hi,b);if(lo>hi)return false;
  }return true;
 }
 void ray(V o,V d,int64_t& id,double& z,double* weights){
  id=-1;z=INFINITY;weights[0]=weights[1]=weights[2]=NAN;
  int stack[128],top=0;stack[top++]=0;double dn=std::sqrt(dot(d,d));
  while(top){const Node& n=nodes[stack[--top]];if(!box(n,o,d,z))continue;
   if(n.left>=0){stack[top++]=n.right;stack[top++]=n.left;continue;}
   for(int i=n.begin;i<n.end;i++){
    const Face& a=f[order[i]];V h=cross(d,a.e2);double det=dot(a.e1,h);
    if(std::abs(det)<=1e-13*std::max(a.scale*dn,1e-30))continue;
    V s=sub(o,a.p),q=cross(s,a.e1);double u=dot(s,h)/det,v=dot(d,q)/det,t=dot(a.e2,q)/det;
    if(u < -1e-10||v < -1e-10||u+v > 1+1e-10||t<=1e-5)continue;
    if(t<z||(t==z&&a.id<id)){z=t;id=a.id;weights[0]=1-u-v;weights[1]=u;weights[2]=v;}
   }
  }
 }
};
extern "C" {
void* make_scene(const double* verts,const int64_t* tris,int64_t count){
 Scene* s=new Scene;s->f.reserve(count);s->order.reserve(count);
 for(int64_t i=0;i<count;i++){
  V p[3];for(int k=0;k<3;k++)for(int j=0;j<3;j++)p[k][j]=verts[3*tris[3*i+k]+j];
  Face f;f.p=p[0];f.e1=sub(p[1],p[0]);f.e2=sub(p[2],p[0]);f.id=i;f.scale=std::sqrt(dot(f.e1,f.e1)*dot(f.e2,f.e2));
  for(int j=0;j<3;j++){double low=std::min({p[0][j],p[1][j],p[2][j]}),high=std::max({p[0][j],p[1][j],p[2][j]});
   double pad=4e-10*(1+std::abs(low)+std::abs(high)+high-low);f.lo[j]=low-pad;f.hi[j]=high+pad;f.center[j]=(low+high)*.5;}
  s->f.push_back(f);s->order.push_back(i);
 }s->build(0,count);return s;
}
void trace(void* scene,const double* origin,const double* dirs,int64_t count,int64_t* ids,double* depth,double* weights){
 Scene* s=(Scene*)scene;V o={{origin[0],origin[1],origin[2]}};
 for(int64_t i=0;i<count;i++){V d={{dirs[3*i],dirs[3*i+1],dirs[3*i+2]}};s->ray(o,d,ids[i],depth[i],weights+3*i);}
}
void free_scene(void* s){delete (Scene*)s;}
}
