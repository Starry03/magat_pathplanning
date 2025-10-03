#!.venv/bin/python

import os
import scipy.io as sio
import numpy as np

root = "./offlineExpert/trainset/map20x20_density_p1/10_Agent/train"
path = os.path.join(root, sorted(os.listdir(root))[0])
mat = sio.loadmat(path)
print("File:", os.path.basename(path))
for k,v in mat.items():
    if k.startswith("__"): continue
    arr = np.array(v)
    print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")