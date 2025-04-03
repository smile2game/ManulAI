"""
手动实现 flash attn v1
思路是 

Bandwidth:
SRAM = 19TB/s (size = 20 MB) 
HBM = 1.5TB/s (size = 40GB)
===========================================================================================
Normal Attn:
Q (N,d)
K (N,d)
V (N,d)
其中 N >> d

S = Q@K (N,N)
P = softmax(S) (N,N)
S,P on the HBM need O(N**2) 

Algo0 :
load Q,K by blocks from  HBM,compute S = Q@K, write S to HBM
read S from HBM,compute P = softmax(S),write P to HBM
load P and V by blocks from HBM,compute O = P@V,write O to HBM
return O
===========================================================================================

flash attn v1
-----------------------------------------------------------------------------------------
softmax:
softmax(x_j) = exp(x_j) / sum(exp(x_i) )  
>> row_wise逐行算,用到平铺算法
-----------------------------------------------------------------------------------------
safe_softmax:
m(x) = max(x)
f(x) = [ exp(x_1 - m(x))   ... exp(x_j - m(x)) ]
l(x) = sum(f(x))
safe_softmax = f(x) / l(x)
-----------------------------------------------------------------------------------------
blocked_safe_softmax:
关键点在于 每次只有blocked的数据,如何求解全局的 softmax
x = [x1,x2]
m(x) = max(x1,x2)
f(x) = f([x1,x2]) = [exp(m(x1)- m(x))*f(x1) , exp(m(x2)- m(x))*f(x2)]
l(x) = l([x1,x2]) = exp(m(x1)- m(x))*l(x1) + exp(m(x2)- m(x))*l(x2)

"""

"""
exp(x) 下凸曲线,g(exp(x)) = exp(x),容易上溢
"""
import torch 
# x = torch.arange(16).reshape(4,4)
x = torch.arange(4)
print(f"x is {x}")

#普通softmax
x_softmax = (x).exp()/(x).exp().sum()
print(f"x_softmax is {x_softmax}\n")

"""
safe softmax 同时除以全局最大的exp(max(x)),也就是在指数部分减去 x_max
"""
x_max = x.max()
x_safe_softmax = (x-x_max).exp()/(x-x_max).exp().sum()
print(f"x_safe_softmax is {x_safe_softmax}\n")

"""
online softmax, 先算好N个数的softmax,再追加一个 x_{i+1}
"""
x_pre = x[:-1]
x_max_pre = x_pre.max()
x_sum_pre = (x_pre-x_max_pre).exp().sum()

x_max_cur = torch.max(x_max_pre,x[-1])
x_sum_cur = x_sum_pre * torch.exp(x_max_pre - x_max_cur) + torch.exp(x[-1] - x_max_cur)  #之前减去的不是全局最大值 

x_online_softmax = torch.exp(x-x_max_cur) / x_sum_cur
print(f"x_online_softmax is {x_online_softmax}")
assert torch.allclose(x_safe_softmax, x_online_softmax)

"""
block online softmax >> 改写为 for循环去做 
"""
x_blocks = torch.split(x,split_size_or_sections=2,dim = 0)
# print(f"x_block is {x_block}")

# x_max_block0 = x_block[0].max()
# x_sum_block0 = torch.exp(x_block[0] - x_max_block0).sum()

# x_max_block1 = x_block[1].max()
# x_sum_block1 = torch.exp(x_block[1] - x_max_block1).sum() #这块其实是用不到的?
x_max_old = torch.tensor(0.0)
x_sum_old = torch.tensor(0.0)
for x_block in x_blocks:
    x_max_block = x_block.max()
    x_max_new = torch.max(x_max_old,x_max_block)
    x_sum_new = x_sum_old * torch.exp(x_max_old - x_max_new) + torch.exp(x_block - x_max_new).sum()
    x_max_old = x_max_new
    x_sum_old = x_sum_new

# x_max_global = torch.max(x_max_block0,x_max_block1)
# x_sum_global = x_sum_block0 * torch.exp(x_max_block0 - x_max_global) + x_sum_block1 * torch.exp(x_max_block1 - x_max_global)
print(f"x is {x},x_max_new is {x_max_new},x_sum_new is {x_sum_new}")
x_block_online_softmax  = torch.exp(x - x_max_old)/x_sum_old
print(f"x_block_online_softmax is {x_block_online_softmax}")
assert torch.allclose(x_block_online_softmax,x_softmax)


"""
batch online softmax >> 改写为 for循环去做 
"""
x = torch.arange(16).reshape(4,4)



assert torch.allclose(x_batch_online_softmax[0],x_softmax)