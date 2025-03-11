准备工作

```python
#clone
git clone https://ghfast.top/https://github.com/smile2game/mnist-dits.git
#免密
git config --global credential.helper store
```



# 3.10 手撕模型

## debug看下

train流程

```python
from config import *
#数据
dataset = MNIST()
BATCH_SIZE = 2000
dataloader = DataLoader(dataset,batch_size = BATCH_SIZE,shuffle = True, num_workers = 10,persistent_workers = True)

#模型
model =  DiT(img_size = 28,patch_size = 4,channel = 1,emb_size = 64,label_num = 10, dit)
model.train() #切换模式
#优化器 
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
loss_fn = nn.L1Loss()

#训练
EPOCH = 500
iter_cnt = 0
for epoch in range(EPOCH):
    for imgs,label in dataloader:
        #数据预处理
        x = imgs * 2 - 1 #像素变换 [0,1] >> [-1,1]，符合噪音的高斯分布
        t = torch.randint(0,T,((imgs.size(0)),)) #每张图片随机的 t 时刻
        y = labels 
        
        #加噪与预测 
        x,noise = forward_add_noise(x,t)
        pred_noise = model(x.to(DEVICE), y.to(DEVICE),t.to(DEVICE))
        
        #梯度降
        loss = loss_fn(fred_noise,noise.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iter_cnt % 1000 == 0:
            torch.save(model.state_dict(),f"model_{iter_cnt}.pth")
        iter_cnt +=1 

```



## 学到的知识：

数据集 \_\_init\_\_

```python
from torch.util.data import Dataset 
from torchvision.transforms.v2 import PILToTensor,Compose

class MNIST(Dataset)

```

模型 \_\_init\_\_

```python
class DiT(nn.Module):
    def __init__(self,img_size,patch_size,channel,emb_size, label_num,dit_num, head):
        
```

加噪 forward\_add\_noise

```python
betas = torch.linspace(0.0001.0.02,T) #噪声方差，前期小后期大 (T,）
alphas = 1-betas #保留原始信号的 比例 
alphas_cumprod = torch.cumprod(alphas,dim = -1) #alpha_t累乘 (T,)
alphas_cumprod_prev = torch.cat(torch.tensor([1.0]),alphas_cumprod[:-1]） ,dim =-1) #alpha_t-1累乘 (T,),整体后移一位，前面补个1。记作前一个时间步的累积
variance = (1- alphas)* 

def forward_add_noise(x,t):
    noise = torch.randn_like(x)
    batch_alphas_cumprod = 
    
```









# 3.11 手撕框架&#x20;

## 手撕ddp



## Deepspeed训练&#x20;



## Megatron训练&#x20;



## Accelerate训练



# 3.12手撕 并行&#x20;

## Odysseus

