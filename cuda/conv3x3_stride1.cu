//stride -1 basic convolution code
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void conv3x3(float *in, float *ker, float *out, int N, int K) {
    int OUT = N - K + 1;
    int tid = threadIdx.x; 

    int r = tid / OUT;  
    int c = tid % OUT;  
    if (r < OUT && c < OUT) {
        float sum = 0;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                sum += in[(r+i)*N + (c+j)] * ker[i*K + j];
            }
        }
        out[r*OUT + c] = sum;
    }
}

int main() {
    int N = 5; 
    int K = 3;  
    int OUT = N - K + 1;

    float input[25] = {   
        1,2,3,4,5,
        6,7,8,9,10,
        11,12,13,14,15,
        16,17,18,19,20,
        21,22,23,24,25
    };

    float kernel[9] = {   
        1,0,1,
        0,1,0,
        1,0,1,
    };

    float output[9];

    float *din, *dker, *dout;
    cudaMalloc(&din, N*N*sizeof(float));
    cudaMalloc(&dker, K*K*sizeof(float));
    cudaMalloc(&dout, OUT*OUT*sizeof(float));

    cudaMemcpy(din, input, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dker, kernel, K*K*sizeof(float), cudaMemcpyHostToDevice);
    conv3x3<<<1, OUT*OUT>>>(din, dker, dout, N, K);

    cudaMemcpy(output, dout, OUT*OUT*sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Output:" << endl;
    for (int i = 0; i < OUT; i++) {
        for (int j = 0; j < OUT; j++) {
            cout << output[i*OUT + j] << " ";
        }
        cout << endl;
    }

    cudaFree(din);
    cudaFree(dker);
    cudaFree(dout);

    return 0;
}
