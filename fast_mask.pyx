cimport numpy as np
import numpy as npy
from libc.stdint cimport uint8_t
from libcpp cimport bool

cdef extern from "fast_mask_lib.hpp":
    # 导入C++中的函数声明
    void multiply_cpp(char* img_NHW_ptr, bool* mask_KNHW_ptr, char* output_ptr,
                   int K, int N, int H, int W, int mode)


def multiply_py(np.ndarray[uint8_t, ndim=3] img_NHW, np.ndarray[char, ndim=4] mask_KNHW, mode:int = 0):
    # 提取2D数组的指针列表
    cdef bool* mask_KNHW_ptr = <bool*> mask_KNHW.data
    cdef char* img_NHW_ptr = <char*> img_NHW.data
    cdef int K = mask_KNHW.shape[0]
    cdef int N = mask_KNHW.shape[1]
    cdef int H = mask_KNHW.shape[2]
    cdef int W = mask_KNHW.shape[3]
    assert img_NHW.shape[0]==N and img_NHW.shape[1]==H and img_NHW.shape[2]==W
    cdef np.ndarray[char, ndim=4] output = npy.empty((K, N, H, W), dtype=npy.uint8)
    cdef char* output_ptr = <char*> output.data

    # 调用C++函数处理数组
    multiply_cpp(img_NHW_ptr, mask_KNHW_ptr,  output_ptr,
                  K, N, H, W, mode)
    return output
    