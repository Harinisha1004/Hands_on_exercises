// stride -1 padding -1 
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void conv2d(const float *in, const float *ker, float *out,
                       int N, int K, int stride, int padding) {

    int OUT = (N - K + 2 * padding) / stride + 1;

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < OUT && c < OUT) {
        float sum = 0.0f;

        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                int in_r = r * stride + i - padding;
                int in_c = c * stride + j - padding;

                if (in_r >= 0 && in_r < N && in_c >= 0 && in_c < N) {
                    sum += in[in_r * N + in_c] * ker[i * K + j];
                }
            }
        }
        out[r * OUT + c] = sum;
    }
}

int main() {
    int N = 5; 
    int K = 3; 
    int stride = 1;
    int padding = 1;

    int OUT = (N - K + 2 * padding) / stride + 1;

    float input[25] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    float kernel[9] = {
        1, 0, 1,
        0, 1, 0,
        1, 0, 1
    };

    float *din, *dker, *dout;
    cudaMalloc(&din, N * N * sizeof(float));
    cudaMalloc(&dker, K * K * sizeof(float));
    cudaMalloc(&dout, OUT * OUT * sizeof(float));

    cudaMemcpy(din, input, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dker, kernel, K * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16); 
    dim3 blocks((OUT + threads.x - 1) / threads.x,
                (OUT + threads.y - 1) / threads.y);

    conv2d<<<blocks, threads>>>(din, dker, dout, N, K, stride, padding);
    cudaDeviceSynchronize();

    float *output = new float[OUT * OUT];
    cudaMemcpy(output, dout, OUT * OUT * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Output:" << endl;
    for (int i = 0; i < OUT; i++) {
        for (int j = 0; j < OUT; j++) {
            cout << output[i * OUT + j] << " ";
        }
        cout << endl;
    }

    cudaFree(din);
    cudaFree(dker);
    cudaFree(dout);

    return 0;
}
