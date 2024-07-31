import torch
from torch import nn

class SMAPE(torch.nn.Module):
    
    def __init__(self, eps=0.01):
        super(SMAPE, self).__init__()
        self.eps = eps
    
    def forward(self, pred, gt):
        loss = torch.mean(torch.abs(pred - gt) / (torch.abs(pred) + torch.abs(gt) + self.eps))
        return loss

class MLTDLoss(torch.nn.Module):
    
    def __init__(self):
        super(MLTDLoss, self).__init__()
        self.criterion = SMAPE()
        self.temporal_weight = 0.5
        
    def forward(self, preds, gt):
        
        img_preds = preds[:, :, :3]
        sequence_length = preds.shape[1]
        single_loss = []
        temporal_loss = []
        
        # Single loss
        for i in range(sequence_length): 
            single_loss.append(self.criterion(img_preds[:, i], gt[:, i]))

        # Temporal loss
        for i in range(1, sequence_length):
            temp_img_preds = img_preds[:, i] - img_preds[:, i - 1]
            temp_gt = gt[:, i] - gt[:, i - 1]
            temporal_loss.append(self.criterion(temp_img_preds, temp_gt))

        single_loss = torch.mean(torch.stack(single_loss))
        temporal_loss = torch.mean(torch.stack(temporal_loss))
        total_loss = single_loss + self.temporal_weight * temporal_loss
        
        return total_loss

class RAELoss(torch.nn.Module):
    
    def __init__(self):
        super(RAELoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.temporal_weight = 0.2
        
    def forward(self, preds, gt):
        
        img_preds = preds[:, :, :3]
        sequence_length = preds.shape[1]
        single_loss = []
        temporal_loss = []
        
        # We don't use gradient loss here because it doesn't give a good result
        
        # Single loss
        for i in range(sequence_length): 
            single_loss.append(self.criterion(img_preds[:, i], gt[:, i]))

        # Temporal loss
        for i in range(1, sequence_length):
            temp_img_preds = img_preds[:, i] - img_preds[:, i - 1]
            temp_gt = gt[:, i] - gt[:, i - 1]
            temporal_loss.append(self.criterion(temp_img_preds, temp_gt))

        single_loss = torch.mean(torch.stack(single_loss))
        temporal_loss = torch.mean(torch.stack(temporal_loss))
        total_loss = single_loss + self.temporal_weight * temporal_loss
        
        return total_loss
    
class KPCNLoss(torch.nn.Module):
    
    def __init__(self):
        super(KPCNLoss, self).__init__()
        self.criterion = nn.L1Loss()
    
    def forward(self, preds, gt):
        
        single_loss = []
        sequence_length = preds.shape[1]
        
        for i in range(sequence_length):
            single_loss.append(self.criterion(preds[:, i], gt[:, i]))
        
        single_loss = torch.mean(torch.stack(single_loss))
        return single_loss

class NTASDLoss(torch.nn.Module):
    
    def __init__(self):
        super(NTASDLoss, self).__init__()
        self.criterion = nn.L1Loss()
        
    def forward(self, preds, gt):
        
        img_preds = preds[:, :, :3]
        
        # NTASD only considers the last frame
        
        # Single loss
        single_loss = self.criterion(img_preds[:, -1], gt[:, -1])
        
        # Temporal loss
        temp_img_preds = img_preds[:, -1] - img_preds[:, -2]
        temp_gt = gt[:, -1] - gt[:, -2]
        temporal_loss = self.criterion(temp_img_preds, temp_gt)
        
        single_loss = torch.mean(single_loss)
        temporal_loss = torch.mean(temporal_loss)
        total_loss = single_loss + temporal_loss
        
        return total_loss

class IDANFLoss(torch.nn.Module):
    
    def __init__(self):
        super(IDANFLoss, self).__init__()
        self.criterion = SMAPE()
        
    def forward(self, preds, gt, alphas):
        
        single_loss = []
        temporal_loss = []
        sequence_length = preds.shape[1]
        
        # Single Loss
        for i in range(sequence_length):
            single_loss.append(self.criterion(preds[:, i], gt[:, i]) + 1e-5 * torch.mean(alphas[:, i]))
        
        # Temporal Loss
        for i in range(1, sequence_length):
            temp_img_preds = preds[:, i] - preds[:, i - 1]
            temp_gt = gt[:, i] - gt[:, i - 1]
            temporal_loss.append(self.criterion(temp_img_preds, temp_gt))
        
        single_loss = torch.mean(torch.stack(single_loss))
        temporal_loss = torch.mean(torch.stack(temporal_loss))
        total_loss = single_loss + 0.25 * temporal_loss
        return total_loss