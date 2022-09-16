import sys

sys.path.append('..')
import torch
import numpy as np
a = np.array([[[1, 2, 3], [4, 5, 6]]])
   # 返回值为6
b = torch.from_numpy(a)
print(b.size(0))
print(b.size(-1))
