from torch import nn
from resnet1d import BaseNet
from tfr import Transformer
import torch

def sleep_model(n_channels, input_size_samples, n_dim = 256):
    class attention(nn.Module):
        
        def __init__(self, n_dim):
            super(attention,self).__init__()
            self.att_dim = n_dim
            self.W = nn.Parameter(torch.randn(n_dim, self.att_dim))
            self.V = nn.Parameter(torch.randn(self.att_dim, 1))
            self.scale = self.att_dim**-0.5
            
        def forward(self,x):
            x = x.permute(0, 2, 1)
            e = torch.matmul(x, self.W)
            e = torch.matmul(torch.tanh(e), self.V)
            e = e*self.scale
            n1 = torch.exp(e)
            n2 = torch.sum(torch.exp(e), 1, keepdim=True)
            alpha = torch.div(n1, n2)
            x = torch.sum(torch.mul(alpha, x), 1)
            return x
        
    class encoder(nn.Module):

        def __init__(self, n_channels, n_dim):
            super(encoder,self).__init__()
            self.model = BaseNet(input_channel = n_channels)
            self.attention = attention(n_dim)
            
        def forward(self, x): 
            x = self.model(x)
            x = self.attention(x)
            return x
        
    class Net(nn.Module):
        
        def __init__(self, n_channels, n_dim):
            super().__init__()
            self.enc = encoder(n_channels, n_dim)
            self.n_dim = n_dim
            
            self.proj = nn.Sequential(
                nn.Linear(self.n_dim, self.n_dim // 2, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(self.n_dim // 2, self.n_dim // 2, bias=True),
            )
            
            self.tfr = Transformer(256, 4, 4, 256, dropout=0.1)
            
            self.tfr_proj = nn.Sequential(
                nn.Linear(self.n_dim, self.n_dim // 2, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(self.n_dim // 2, self.n_dim // 2, bias=True),
            )
            
           
        def forward(self, x, proj='mid'): # B, 7, 1, 3000
            
            if proj=='mid':
                assert len(x.shape) == 3, 'Send a single epoch'
                return self.enc(x[:,:1,:]) # B,256
        
            epoch_len = x.shape[1] # 7          
            temp = []
            
            for i in range(epoch_len):
                temp.append(self.enc(x[:,i,:,:]))
            x = torch.stack(temp, dim=1) # B, 7, 256
            del temp    
       
            if proj=='proj':
                tfr_x = self.tfr(x)
                return self.proj(x[:, epoch_len//2, :]), self.tfr_proj(tfr_x)
            else:
                assert False, 'proj must be mid, proj or pred'
                         
    return Net(n_channels, n_dim)
