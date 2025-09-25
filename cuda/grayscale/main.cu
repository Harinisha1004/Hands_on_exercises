#include <cuda_runtime.h>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

__global__ void grayscale(unsigned char *p,unsigned char *o,int totalPixels){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<totalPixels){
        o[tid]=255-p[tid];
    }
}

int main(){
    Mat img=imread("orange.jpg",IMREAD_GRAYSCALE);
    if(img.empty()){
        cout<<"Failed to load image!"<<endl;
        return -1;
    }else {
        cout<<"Image is present"<<endl;
    }

    cout << "Image size: " << img.rows << "x" << img.cols << endl;
    
    int totalPixels=img.rows*img.cols;
    size_t bytes=totalPixels*sizeof(unsigned char);

    unsigned char *p,*o;

    // for (int i = 0; i < img.rows; i++) {
    //     for (int j = 0; j < img.cols; j++) {
    //         // Access grayscale pixel at (i,j)
    //         p= img.at<uchar>(i, j);
    //         cout << (int)pixel << " ";  // cast to int to print numeric value
    //     }
    //     cout << endl;
    // }
    
    cudaMallocHost((void**)&p,bytes);
    cudaMallocHost((void**)&o,bytes);

    memcpy(p,img.data,bytes);
    unsigned char *da,*db;
    cudaMalloc(&da,bytes);
    cudaMalloc(&db,bytes);
    
    cudaMemcpy(da,p,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(db,o,bytes,cudaMemcpyHostToDevice);

    int threads=256;
    int blocks=(totalPixels+threads-1)/threads;

    grayscale<<<blocks,threads>>>(da,db,totalPixels);
    cudaMemcpy(o,db,bytes,cudaMemcpyDeviceToHost);
    
    Mat output(img.rows,img.cols,CV_8UC1,o);
    imwrite("D:/Harini/CMAKE_PRACTICE/grayscale/build/Debug/output.jpg",output);

    cout<<"Inverted image saved"<<endl;

    cudaFree(da);
    cudaFree(db);
    cudaFreeHost(p);
    cudaFreeHost(o);

    return 0;
};

