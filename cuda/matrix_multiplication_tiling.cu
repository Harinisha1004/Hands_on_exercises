// matrix multiplication with tiling
#include <stdio.h>
#include <cstdlib>
#include <cassert>
#include <iostream>
using namespace std;
#define SHMEM_SIZE (16*16)

__global__ void MatrixMulTiled(int *a,int *b,int *c,int N){
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;

    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int dim=blockDim.x; // or blockDim.y

    int tmp=0;
    for(int i=0;i<(N+dim-1)/dim;i++){
        A[ty*dim+tx]=a[(row*N)+(i*dim)+tx];
        B[ty*dim+tx]=b[(i*dim*N)+(ty*N)+col];
        __syncthreads();

        for(int j=0;j<dim;j++){
            tmp+=A[ty*dim+j]*B[j*dim+tx];
        }
        __syncthreads();
        c[row*N+col]=tmp;
    }
}

void init_array(int *a,int N){
    for(int i=0;i<N*N;i++){
        a[i]=rand()%100;
    }
}

void checker(int *a,int *b,int *c,int N){
    int tmp;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            tmp=0;
            for(int k=0;k<N;k++){
                tmp+=a[i*N+k]*b[k*N+j];
            }
            assert(tmp==c[i*N+j]);
        }
    }
}

int main(){
 int N=1024;
 size_t bytes=N*N*sizeof(int);

 int *a,*b,*c;
 cudaMallocHost(&a,bytes);
 cudaMallocHost(&b,bytes);
 cudaMallocHost(&c,bytes);

 init_array(a,N);
 init_array(b,N);

 int *da,*db,*dc;
 cudaMalloc(&da,bytes);
 cudaMalloc(&db,bytes);
 cudaMalloc(&dc,bytes);

 cudaMemcpy(da,a,bytes,cudaMemcpyHostToDevice);
 cudaMemcpy(db,b,bytes,cudaMemcpyHostToDevice);

 int threads=16;
 int blocks=(N+threads-1)/threads;

 dim3 THREADS(threads,threads);
 dim3 BLOCKS(blocks,blocks);
 MatrixMulTiled<<<BLOCKS,THREADS>>>(da,db,dc,N);

 cudaMemcpy(c,dc,bytes,cudaMemcpyDeviceToHost);
 checker(a,b,c,N);

 cout<<"Program ended successfully";
 return 0;

}
