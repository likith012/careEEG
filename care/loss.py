import torch
import torch.nn.functional as F

# Loss function
class loss_fn(torch.nn.modules.loss._Loss):
    def __init__(self, device, T=0.5):
       
        super(loss_fn, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive):

        # L2 normalize
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)

        # positive logits: Nxk, negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        loss = - l_pos.mean()
                
        return loss

