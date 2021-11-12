import torch
import torch.nn as nn
from .gather import GatherLayer

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, device, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.world_size = world_size
    
    def forward(self, z_i, z_j, dist_labels):
        
        N = 2 * z_i.shape[0] * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        dist_labels = torch.cat((dist_labels, dist_labels),dim=0)
        
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)
            dist_labels = torch.cat(GatherLayer.apply(dist_labels), dim=0)
        
        # calculate similarity and divide by temperature parameter
        z = nn.functional.normalize(z, p=2, dim=1)
        sim = torch.mm(z, z.T) / self.temperature
        dist_labels = dist_labels.cpu()
        
        positive_mask = torch.mm(dist_labels.to_sparse(), dist_labels.T)
        positive_mask = positive_mask.fill_diagonal_(0).to(sim.device)
        zero_diag = torch.ones((N, N)).fill_diagonal_(0).to(sim.device)
        
        # calculate normalized cross entropy value
        positive_sum = torch.sum(positive_mask, dim=1)
        denominator = torch.sum(torch.exp(sim)*zero_diag,dim=1)
        loss = torch.mean(torch.log(denominator) - \
                          (torch.sum(sim * positive_mask, dim=1)/positive_sum))
        
        return loss
