// dot product with padding: SIMD vs Scalar
#include <iostream>
#include <chrono>
#include <immintrin.h>
using namespace std;

int dot_product_simd(const int32_t* A, const int32_t* B, int padded_value) {
    __m256i sum = _mm256_setzero_si256();

    for (int i = 0; i < padded_value; i += 8) {
        __m256i veca = _mm256_load_si256((__m256i*)(A + i));
        __m256i vecb = _mm256_load_si256((__m256i*)(B + i));
        __m256i product = _mm256_mullo_epi32(veca, vecb);
        sum = _mm256_add_epi32(sum, product);
    }

    __m128i lower = _mm256_extracti128_si256(sum, 0);
    __m128i upper = _mm256_extracti128_si256(sum, 1);
    __m128i final_value = _mm_add_epi32(lower, upper);
    final_value = _mm_hadd_epi32(final_value, final_value);
    final_value = _mm_hadd_epi32(final_value, final_value);

    return _mm_cvtsi128_si32(final_value);
}

int dot_product_scalar(const int32_t* A, const int32_t* B, int count) {
    int sum = 0;
    for (int i = 0; i < count; ++i) {
        sum += A[i] * B[i];
    }
    return sum;
}

int main() {
    int elements = 25072;
    int padded_element = ((elements + 7) / 8) * 8;

    alignas(32) int32_t A[padded_element] = {0};
    alignas(32) int32_t B[padded_element] = {0};

    for (int i = 0; i < elements; ++i) {
        A[i] = i + 1;
        B[i] = i + 2;
    }

    double simd_time_taken = 0.0, scalar_time_taken = 0.0;
    int no_times = 12000;
    using namespace std::chrono;

    int simd_result = 0, scalar_result = 0;

    for (int i = 0; i < no_times; ++i) {
        auto start = high_resolution_clock::now();
        simd_result = dot_product_simd(A, B, padded_element);
        auto end = high_resolution_clock::now();
        duration<double, std::micro> dt = end - start;
        simd_time_taken += dt.count();
    }
    double simd_avg_time = simd_time_taken / no_times;
    cout << endl << "The total time for (SIMD) " << no_times << " runs is: " << simd_avg_time << " duration";

    for (int i = 0; i < no_times; ++i) {
        auto start = high_resolution_clock::now();
        scalar_result = dot_product_scalar(A, B, elements);
        auto end = high_resolution_clock::now();
        duration<double, std::micro> dt = end - start;
        scalar_time_taken += dt.count();
    }
    double scalar_avg_time = scalar_time_taken / no_times;
    cout << endl << "The total time for (SCALAR) " << no_times << " runs is: " << scalar_avg_time << " duration" << endl;

    cout << "The results are: ";
    if (simd_result == scalar_result) {
        cout << endl << "SIMD Result: " << simd_result << endl;
        cout << "Scalar Result: " << scalar_result << endl;
    } else {
        cout << endl << "Mismatch Detected!" << endl;
        cout << "SIMD Result: " << simd_result << ", Scalar Result: " << scalar_result << endl;
    }

    return 0;
}
