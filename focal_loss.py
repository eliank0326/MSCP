import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0,alpha=None,reduction='mean'):
        """
        gamma: 调节难易样本权重的指数系数
        alpha: 每个类别的权重（list or Tensor），如 [0.25, 0.25, 0.25, 0.25]
        reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, C]，softmax前的logits
        targets: [B]，ground truth class indices
        """
        log_probs = F.log_softmax(inputs,dim=1)
        probs = torch.exp(log_probs) #[B,C]
        targets = targets.view(-1,1)

        pt = probs.gather(1, targets).clamp(min=1e-6) #避免log(0)
        log_pt = log_probs.gather(1, targets)

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets.squeeze()]
            log_pt = log_pt * alpha_t.to(inputs.device)

        loss = -((1-pt)**self.gamma) * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss #none

