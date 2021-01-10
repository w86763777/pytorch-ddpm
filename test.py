import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, v=0):
        print(x.shape)
        return (x - v).mean()


m = Model().cuda()
m = nn.DataParallel(m)

x = torch.ones((4, 8))
v = 5
print(m(x, 2))
