// 8x8 Matrix Transpose: SIMD vs Scalar
#include <iostream>
#include <chrono>
#include <immintrin.h>
using namespace std;

// SIMD version using AVX2 intrinsics (1D array)
void simd_matrix_transpose_1d(const int32_t A[64],int32_t B[64]){
    __m256i rows[8];
    for(int i=0;i<8;i++){
        rows[i]=_mm256_loadu_si256((__m256i*)&A[i*8]);
    }

    __m256i t[8];
    for(int i=0;i<8;i=i+2){
        t[i]=_mm256_unpacklo_epi32(rows[i],rows[i+1]);
        t[i+1]=_mm256_unpackhi_epi32(rows[i],rows[i+1]);
    }
    __m256i s[8];
    for(int i=0;i<8;i=i+4){
        s[i]=_mm256_unpacklo_epi64(t[i],t[i+2]);
        s[i+1]=_mm256_unpackhi_epi64(t[i],t[i+2]);
        s[i+2]=_mm256_unpacklo_epi64(t[i+1],t[i+3]);
        s[i+3]=_mm256_unpackhi_epi64(t[i+1],t[i+3]);
    }

    for(int i=0;i<4;i++){
        rows[i]=_mm256_permute2x128_si256(s[i],s[i+4],0x20);
    }
    for(int i=0;i<4;i++){
        rows[i+4]=_mm256_permute2x128_si256(s[i],s[i+4],0x31);
    }

    for(int i=0;i<8;i++){
        _mm256_storeu_si256((__m256i*)&B[i*8],rows[i]);
    }

}

// Scalar version (1D array)
void scalar_matrix_transpose_1d(const int32_t A[64],int32_t B[64]){
    int col=8,row=8;
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            B[j*row+i]=A[i*col+j];
        }
    }
}

int main(){
    int row=8,col=8,value=0;
    int32_t one_d[row*col],simd_result[row*col],scalar_result[row*col];
    for(int i=0;i<row*col;i++){
        one_d[i]=i;
    }

    for(int i=0;i<row*col;i++){
        if(i%8==0){
            cout<<endl;
        }
        cout<<one_d[i]<<" ";
    }

    double simd_time_taken=0.0f,scalar_time_taken=0.0f;
    int no_times=100000;
    using namespace std::chrono;
    for(int i=0;i<no_times;i++){
        auto start=high_resolution_clock::now();
        simd_matrix_transpose_1d(one_d,simd_result);
        auto end=high_resolution_clock::now();
        duration<double,std::nano> dt=end-start;
        simd_time_taken+=dt.count();
    }
    double simd_average_time=simd_time_taken/no_times;
    cout<<endl<<"The total time for (SIMD) "<<no_times<<" runs is: "<<simd_average_time<<" duration";
    
    for(int i=0;i<no_times;i++){
        auto start=high_resolution_clock::now();
        scalar_matrix_transpose_1d(one_d,scalar_result);
        auto end=high_resolution_clock::now();
        duration<double,std::nano> dt=end-start;
        scalar_time_taken+=dt.count();
    }
    double scalar_average_time=scalar_time_taken/no_times;
    cout<<endl<<"The total time for (SCALAR) "<<no_times<<" runs is: "<<scalar_average_time<<" duration"<<endl;
    
    cout<<"The values are:";
    bool mismatch=false;
    for(int i=0;i<64 && !mismatch;i++){
        if(i%8==0){
            cout<<endl;
        }
            if(simd_result[i]==scalar_result[i]){
                cout<<simd_result[i]<<" ";
            }else{
                cout<<endl<<"At the index:("<<i<<") is a mismatch";
                mismatch=true;
                break;
            }

            
    }

    return 0;
}