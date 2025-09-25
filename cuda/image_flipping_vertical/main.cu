#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

__global__ void vertical_flipping_top_down(unsigned char *a,unsigned char *b,int totalPixels,int img_cols,int img_rows){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<totalPixels){
        int row = tid / img_cols;
        int col = tid % img_cols;
        int flipped_row = (img_rows - 1) - row;

        int in_idx  = (row * img_cols + col) * 3;          
        int out_idx = (flipped_row * img_cols + col) * 3;  

        // Copy BGR values
        b[out_idx]     = a[in_idx];       // B
        b[out_idx + 1] = a[in_idx + 1];   // G
        b[out_idx + 2] = a[in_idx + 2];   // R
    }
}

int main(){
    Mat img=imread("flipping.jpg",IMREAD_COLOR);
    if(img.empty()){
        cout<<"Failed to load the image"<<endl;
        return -1;
    }else{
        cout<<"Image is loaded"<<endl;
    }

    int totalPixels=img.rows*img.cols;
    size_t bytes=3*totalPixels*sizeof(unsigned char);
    cout<<"Image :"<<img.rows<<"x"<<img.cols;
    
    unsigned char *a,*b;
    cudaMallocHost((void**)&a,bytes);
    cudaMallocHost((void**)&b,bytes);

    memcpy(a,img.data,bytes);

    unsigned char *da,*db;
    cudaMalloc((void**)&da,bytes);
    cudaMalloc((void**)&db,bytes);

    cudaMemcpy(da,a,bytes,cudaMemcpyHostToDevice);

    int threads=256;
    int blocks=(totalPixels+threads-1)/threads;
    vertical_flipping_top_down<<<blocks,threads>>>(da,db,totalPixels,img.cols,img.rows);
    cudaMemcpy(b,db,bytes,cudaMemcpyDeviceToHost);

    Mat output(img.rows,img.cols,CV_8UC3,b);
    imwrite("D:/Harini/CMAKE_PRACTICE/image_flipping/build/Debug/output.jpg",output);

    cout<<"Inverted image saved"<<endl;
    return 0;
}
