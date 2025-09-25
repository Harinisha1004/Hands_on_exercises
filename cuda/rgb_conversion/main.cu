#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

__global__ void rgb_conversion(unsigned char *b,unsigned char *g,unsigned char *r,unsigned char *output,int totalPixels){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<totalPixels){
        output[tid*3+0]=255-b[tid];
        output[tid*3+1]=255-g[tid];
        output[tid*3+2]=255-r[tid];
    }
}

int main(){
    Mat image=imread("rgb_image.png",IMREAD_COLOR);
    if(image.empty()){
        cout<<"Failed to load the image!"<<endl;
        return -1;
    }else{
        cout<<"Image is present"<<endl;
    }
    cout<<"Image size: "<<image.rows<<"x"<<image.cols<<endl;

    //the order is BGR not RGB
    int totalPixels=image.rows*image.cols;
    size_t bytes=totalPixels*sizeof(unsigned char);
    unsigned char *b,*g,*r,*out;
    // b=new unsigned char[totalPixels];
    // g=new unsigned char[totalPixels];
    // r=new unsigned char[totalPixels];

    cudaMallocHost(&b,bytes);
    cudaMallocHost(&g,bytes);
    cudaMallocHost(&r,bytes);
    cudaMallocHost(&out,3*bytes);

    int value=0;
    for(int i=0;i<image.rows;i++){
        for(int j=0;j<image.cols;j++){
            Vec3b pixel=image.at<Vec3b>(i,j);
            b[value]=pixel[0];
            g[value]=pixel[1];
            r[value]=pixel[2];
            value++;
        }
    }

    unsigned char *db,*dg,*dr,*doutput;
    cudaMalloc(&db,bytes);
    cudaMalloc(&dg,bytes);
    cudaMalloc(&dr,bytes);
    cudaMalloc(&doutput,3*bytes);

    cudaMemcpy(db,b,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dg,g,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dr,r,bytes,cudaMemcpyHostToDevice);
    
    int threads=256;
    int blocks=(totalPixels+threads-1)/threads;
    rgb_conversion<<<blocks,threads>>>(db,dg,dr,doutput,totalPixels);

    cudaMemcpy(out,doutput,3*bytes,cudaMemcpyDeviceToHost);
    Mat output(image.rows,image.cols,CV_8UC3,out);
    imwrite("D:/Harini/CMAKE_PRACTICE/rgb_conversion/build/Debug/output.jpg",output);

    cout<<"Inverted image saved"<<endl;
    cudaFree(db);
    cudaFree(dg);
    cudaFree(dr);
    cudaFree(b);
    cudaFree(g);
    cudaFree(r);
    cudaFree(out);
    cudaFree(doutput);
    return 0;
}
