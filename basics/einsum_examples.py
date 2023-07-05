import torch

x = torch.rand((2, 3))
print(x)

# Permute tensors
x_ = torch.einsum('ij->ji', x)
print(f'\nPermuted: \n{x_}')

# Summation
x_ = torch.einsum('ij->', x)
print(f'\nSummed:\n{x_:.2f}')

# Column sum
x_ = torch.einsum('ij->j', x)
print(f'\nColumn Summed:\n{x_}')

# Row sum
x_ = torch.einsum('ij->i', x)
print(f'\nRow Summed:\n{x_}')

# Matrix-Vector multiplication
v = torch.rand((1, 3))
x_ = torch.einsum('ij,kj->ik', x, v)
print(f'\nMatrix-Vector multiplication:\n{x_}')

# Matrix-Matrix multiplication
x_ = torch.einsum('ij,kj->ik', x, x)
print(f'\nMatrix-Matrix multiplication:\n{x_}')

# Dot product first row with first row of matrix
x_ = torch.einsum('i,i->', x[0], x[0])
print(f'\nDot product first row:\n{x_}')

# Dot product with matrix
x_ = torch.einsum('ij,ij->', x, x)
print(f'\nDot product Matrix:\n{x_}')

# Element wise multiplication (Hadamard product)
x_ = torch.einsum('ij,ij->ij', x, x)
print(f'\nElement wise multiplication:\n{x_}')

# Outer product
a = torch.rand((3))
b = torch.rand((5))
x_ =  torch.einsum('i,j->ij', a, b)
print(f'\nOuter proudct:\n{x_}')

# Batch matrix multiplication
a = torch.rand((3, 2, 5))
b = torch.rand((3, 5, 3))
x_ =  torch.einsum('ijk,ikl->ijl', a, b)
print(f'\nBatch matrix multiplication:\n{x_}')

# Matrix diagonal
a = torch.rand((3, 3))
x_ = torch.einsum('ii->i', a)
print(f'\nDiagonal:\n{x_}')
print(a)

# Matrix trace (sum of diagonals)
x_ = torch.einsum('ii->', a)
print(f'\nMatrix trace:\n{x_}')