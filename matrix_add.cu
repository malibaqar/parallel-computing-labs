#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
__global__ void add(const float*a,const float*b,float*c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
int main(){const int n=64*64;float *a,*b,*c;CHECK(cudaMallocManaged(&a,n*sizeof(float)));CHECK(cudaMallocManaged(&b,n*sizeof(float)));CHECK(cudaMallocManaged(&c,n*sizeof(float)));for(int i=0;i<n;i++){a[i]=i;b[i]=2*i;}add<<<(n+255)/256,256>>>(a,b,c,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());for(int i=0;i<n;i++)if(c[i]!=3*i)return 2;puts("CPU reference matched");CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(c));}
