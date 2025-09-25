//matrix multiplication with equal row and column
#include <stdio.h>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <cstdlib> 

using namespace std;

__global__ void matrixMul(int *a,int *b,int *c,int N,int total_elements){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;

    // if(row<N && col<N){
    //     printf("Thread (%d,%d)-> row = %d,col=%d\n",threadIdx.y,threadIdx.x,row,col);
    // }

    if(row<N && col<N){
        int tmp=0;
        for(int i=0;i<N;i++){
            tmp+=a[row*N+i]*b[i*N+col];
        }
        c[row*N+col]=tmp;
    }
    
}

void init_array(int *a,int N){
    for(int i=0;i<N;i++){
        a[i]=rand()%100;
    }
}

void checker(int *a,int *b,int *c,int N,int total_elements){
    int tmp=0;
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
    int N=1024; // value is 1024
    int total_elements=N*N;
    size_t bytes=N*N*sizeof(int);
    
    int *a,*b,*c;
    cudaMallocHost(&a,bytes);
    cudaMallocHost(&b,bytes);
    cudaMallocHost(&c,bytes);

    init_array(a,total_elements);
    init_array(b,total_elements);

    int *da,*db,*dc;
    cudaMalloc(&da,bytes);
    cudaMalloc(&db,bytes);
    cudaMalloc(&dc,bytes);

    cudaMemcpy(da,a,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,bytes,cudaMemcpyHostToDevice);

    int threads=16;
    int blocks=(N+threads-1)/threads;

    //2D grid
    dim3 THREADS(threads,threads); //(16,16)
    dim3 BLOCKS(blocks,blocks); //(64,64)

    matrixMul<<<BLOCKS,THREADS>>>(da,db,dc,N,total_elements);
    cudaDeviceSynchronize();

    cudaMemcpy(c,dc,bytes,cudaMemcpyDeviceToHost);

    checker(a,b,c,N,total_elements);
    cout<<"Program executed successfully";
    return 0;
}
