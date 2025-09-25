// pinned memory(cudaMallocHost,cudaMalloc,cudaMemcpy)
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include<cassert>
#include <iostream>
using namespace std;

__global__ void vector_add(int *a,int *b,int *c,int N){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<N){
    c[tid]=a[tid]+b[tid];
    }
}

void init_array(int *a, int N){
    for(int i=0;i<N;i++){
        a[i]=rand()%100;
    }
}

void checker(int *a,int *b,int *c, int N){
    for(int i=0;i<N;i++){
        assert(a[i]+b[i]==c[i]);
    }
    cout<<"Program completed!";
}

int main(){
    int N=1<<20; //2^20 1048576
    size_t bytes=N*sizeof(bytes);

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

    // CTA and grid dimensions
    int threads=256;
    int blocks=(N+threads-1)/threads;

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vector_add<<<blocks,threads>>>(da,db,dc,N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds=0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    cout<<"The total time taken by the add kernel is: "<<milliseconds<<" ms"<<endl;
    cudaMemcpy(c,dc,bytes,cudaMemcpyDeviceToHost);

    checker(a,b,c,N);
}
