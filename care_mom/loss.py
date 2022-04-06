import torch
import torch.nn.functional as F

class loss_fn(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(loss_fn, self).__init__()
        self.T = T
        self.device = device

    def forward(self, out_1, out_2):
        
        # L2 normalize
        out_1 = F.normalize(out_1, p=2, dim=1)
        out_2 = F.normalize(out_2, p=2, dim=1)

        out = torch.cat([out_1, out_2], dim=0) # 2B, 128
        N = out.shape[0]
        
        # Full similarity matrix
        cov = torch.mm(out, out.t().contiguous()) # 2B, 2B
        sim = torch.exp(cov / self.T) # 2B, 2B

        # Negative similarity matrix
        mask = ~torch.eye(N, device = sim.device).bool()
        neg = sim.masked_select(mask).view(N, -1).sum(dim = -1)

        # Positive similarity matrix
        pos = torch.exp(torch.sum(out_1*out_2, dim=-1) / self.T)
        pos = torch.cat([pos, pos], dim = 0) # 2B

        loss = -torch.log(pos / neg).mean()
        
        return loss