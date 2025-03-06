import torch
import numpy as np
_ = torch.manual_seed(0)

W_rank = 2
d, k = 10, 10

W = torch.randn(d, W_rank) @ torch.randn(W_rank, k)
r = np.linalg.matrix_rank(W)

U, S, V = torch.svd(W)
U_r = U[:,:r]
#print(S)
S_r = torch.diag(S[:r])
V_r = V[:,:r].t()
#print(S_r)
B = U_r @ S_r
A = V_r

bias = torch.randn(d)
x = torch.randn(d)

y = W @ x + bias
y_l = (B @ A) @ x + bias

print(y)
print(y_l)
