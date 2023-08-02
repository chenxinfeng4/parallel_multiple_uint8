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


## RGB24 的数据处理如何加速？
---
RGB24 的数据处理过程总， HWC 的格式处理特别慢。是否有合适的 SIMD 加速？
```python
# 3274.33it/s
for _ in tqdm.trange(5000):
    out2 = img_CHW * mask_HW

# 217.13it/s  非常慢
for _ in tqdm.trange(5000):
    out3 = img_HWC * mask_HW[..., None]
```
