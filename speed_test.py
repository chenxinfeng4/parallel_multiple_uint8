# %%
import numpy as np
import tqdm
import os
import fast_mask
import time

# %%
K, N, H, W, C = 2, 9, 800, 1280, 3
img_NHW = np.random.randint(0, 256, size=(N, H, W)).astype(np.uint8)
mask_KNHW = (np.random.rand(K, N, H, W) > 0.5).astype(np.uint8)

out_np_KNHW = img_NHW * mask_KNHW
out_0_KNHW = fast_mask.multiply_py(img_NHW, mask_KNHW, 0)
out_1_KNHW = fast_mask.multiply_py(img_NHW, mask_KNHW, 1)
out_2_KNHW = fast_mask.multiply_py(img_NHW, mask_KNHW, 2)
assert np.all(out_np_KNHW == out_0_KNHW)
assert np.all(out_np_KNHW == out_1_KNHW)
assert np.all(out_np_KNHW == out_2_KNHW)

# %% bench mark 可以1. 查看速度  2.查看CPU的多核占用情况
print('top -p ', os.getpid()) 
for _ in tqdm.trange(5000, desc='Numpy'):
    out_np_KNHW = img_NHW * mask_KNHW

# for loop
for _ in tqdm.trange(5000, desc="C++ for loop"):
    out_0_KNHW = fast_mask.multiply_py(img_NHW, mask_KNHW, 0)

# avx2
for _ in tqdm.trange(5000, desc="C++ AVX2"):
    out_1_KNHW = fast_mask.multiply_py(img_NHW, mask_KNHW, 1)

# avx512
for _ in tqdm.trange(5000, desc="C++ AVX512"):
    out_2_KNHW = fast_mask.multiply_py(img_NHW, mask_KNHW, 2)
