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
out_simd_KNHW = fast_mask.multiply_py(img_NHW, mask_KNHW)
assert np.all(out_np_KNHW == out_simd_KNHW)

# %% bench mark 可以1. 查看速度  2.查看CPU的多核占用情况
print('top -p ', os.getpid()) 
for _ in tqdm.trange(5000):
    out_np_KNHW = img_NHW * mask_KNHW

for _ in tqdm.trange(5000):
    out_simd_KNHW = fast_mask.multiply_py(img_NHW, mask_KNHW)
