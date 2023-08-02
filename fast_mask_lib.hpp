//g++ -mavx2 fast_simd.cpp 

#include <iostream>
#include <immintrin.h>

void multiply_cpp_avx2(char* img_NHW_ptr, bool* mask_NHW_ptr, 
                  char* output_NHW_ptr, int size){
    __m256i one = _mm256_set1_epi8(1);
    __m256i xor_mask = _mm256_set1_epi8(static_cast<char>(0xff));

    const int width = 256 / 8; // =32 uint8 per batch
    size_t batch = size / width;

    uint8_t (* vectorIMG)[width]   = reinterpret_cast<uint8_t(*)[width]>(img_NHW_ptr);
    uint8_t (* vectorMASK)[width]   = reinterpret_cast<uint8_t(*)[width]>(mask_NHW_ptr);
    uint8_t (* vectorResult)[width] = reinterpret_cast<uint8_t(*)[width]>(output_NHW_ptr);

    for(size_t i =0; i< batch; ++i){
        // 使用 AVX2 指令进行矢量操作
        __m256i vec_mask = _mm256_loadu_si256((__m256i*) vectorMASK[i]);
        __m256i vec_img  = _mm256_loadu_si256((__m256i*) vectorIMG[i]);
        __m256i b1 = _mm256_sub_epi8(vec_mask, one);
        __m256i b2 = _mm256_xor_si256(b1, xor_mask); //0 -> 0; 1 -> 0xFF
        // __m256i resultVector = _mm256_and_si256(vec_img, vec_mask);  为了性能测试
        __m256i resultVector = _mm256_and_si256(vec_img, b2); 
        // 将结果存储到结果数组中
        _mm256_storeu_si256((__m256i*) vectorResult[i], resultVector);
    }

    // 暂时不考虑边角料，让数据恰好填充 batch
    // uint8_t * uptrIMG = reinterpret_cast<uint8_t*>(img_NHW_ptr);
    // uint8_t * uptrMASK = reinterpret_cast<uint8_t*>(mask_NHW_ptr);
    // uint8_t * uptrResult = reinterpret_cast<uint8_t*>(output_NHW_ptr);
    // for(size_t i=batch*width; i<size; ++i){
    //     uptrResult[i] = uptrIMG[i] * uptrMASK[i];
    // }
}

void multiply_cpp_avx512(char* img_NHW_ptr, bool* mask_NHW_ptr, 
                  char* output_NHW_ptr, int size)
{
    __m512i one = _mm512_set1_epi8(1);
    __m512i xor_mask = _mm512_set1_epi8(static_cast<char>(0xff));

    const int width = 512 / 8; // =64 uint8 per batch
    size_t batch = size / width;

    uint8_t (* vectorIMG)[width]   = reinterpret_cast<uint8_t(*)[width]>(img_NHW_ptr);
    uint8_t (* vectorMASK)[width]   = reinterpret_cast<uint8_t(*)[width]>(mask_NHW_ptr);
    uint8_t (* vectorResult)[width] = reinterpret_cast<uint8_t(*)[width]>(output_NHW_ptr);

    for(size_t i =0; i< batch; ++i){
        // 使用 AVX512 指令进行矢量操作
        __m512i vec_img  = _mm512_loadu_si512(vectorIMG[i]);
        __m512i vec_mask = _mm512_loadu_si512(vectorMASK[i]);
        __m512i b1 = _mm512_sub_epi8(vec_mask, one);
        __m512i b2 = _mm512_xor_si512(b1, xor_mask);
        // __m512i resultVector = _mm512_and_si512(vec_img, vec_mask); 为了性能测试
        __m512i resultVector = _mm512_and_si512(vec_img, b2);  
        // 将结果存储到结果数组中
        _mm512_storeu_si512(vectorResult[i], resultVector);
    }

    // uint8_t * uptrIMG = reinterpret_cast<uint8_t*>(img_NHW_ptr);
    // uint8_t * uptrMASK = reinterpret_cast<uint8_t*>(mask_NHW_ptr);
    // uint8_t * uptrResult = reinterpret_cast<uint8_t*>(output_NHW_ptr);
    // for(size_t i=batch*width; i<size; ++i){
    //     uptrResult[i] = uptrIMG[i] * uptrMASK[i];
    // }
}


void multiply_cpp_forloop(char* img_NHW_ptr, bool* mask_NHW_ptr, 
                  char* output_NHW_ptr, int size){
    uint8_t * uptrIMG = reinterpret_cast<uint8_t*>(img_NHW_ptr);
    uint8_t * uptrMASK = reinterpret_cast<uint8_t*>(mask_NHW_ptr);
    uint8_t * uptrResult = reinterpret_cast<uint8_t*>(output_NHW_ptr);
    for(int i=0; i<size; ++i){
        uptrResult[i] = uptrIMG[i] * uptrMASK[i];
    }
}

//multiply_cpp_forloop multiply_cpp_avx2 multiply_cpp_avx512
#define MULTIPLY_BY  multiply_cpp_avx2
void multiply_cpp(char* img_NHW_ptr, bool* mask_KNHW_ptr, char* output_ptr,
                   int K, int N, int H, int W, int mode)
{
    int size = N*H*W;
    auto multiply_by = multiply_cpp_forloop;
    if (mode == 1) {
        multiply_by = multiply_cpp_avx2;
    } else if (mode == 2) {
        multiply_by = multiply_cpp_avx512;
    }
    // 开启 openmp 多线程加速
    // #pragma omp parallel for num_threads(3)
    for(int k=0; k<K; k++){
        char* output_NHW_ptr = &output_ptr[k*size];
        bool* mask_NHW_ptr = &mask_KNHW_ptr[k*size];
        multiply_by(img_NHW_ptr, mask_NHW_ptr, 
                  output_NHW_ptr, size);
    }
}
