import torch
# 尝试在GPU上创建一个张量
if torch.cuda.is_available():
    tensor = torch.rand(3, 3).cuda()
    print("Tensor on CUDA:", tensor)
else:
    print("CUDA is not available. Check your installation.")




