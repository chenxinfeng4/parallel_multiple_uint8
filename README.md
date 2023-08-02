# 安装指南
---
```bash
git clone URL
cd GIT_PROJECT

python setup.py build_ext --inplace
```

## 速度测试
```bash
python speed_test.py
```

## 选项：切换不同的乘法版本
打开`fast_mask_lib.hpp`. 修改配置后执行 `python setup.py build_ext --inplace` 。
```cpp
multiply_cpp_forloop  FOR 循环
multiply_cpp_avx512   AVX512
multiply_cpp_avx2     AVX256

#define MULTIPLY_BY  multiply_cpp_***

```

## 选项：是否使用多线程
打开`fast_mask_lib.hpp`. 修改配置后执行 `python setup.py build_ext --inplace` 。
```bash
void multiply_cpp(char* img_NHW_ptr, bool* mask_KNHW_ptr, char* output_ptr,
                   int K, int N, int H, int W)
{
    int size = N*H*W;
    // 开启 openmp 多线程加速
    // #pragma omp parallel for num_threads(3)
    for(int k=0; k<K; k++){
        char* output_NHW_ptr = &output_ptr[k*size];
        bool* mask_NHW_ptr = &mask_KNHW_ptr[k*size];
        MULTIPLY_BY(img_NHW_ptr, mask_NHW_ptr, 
                  output_NHW_ptr, size);
    }
}
```