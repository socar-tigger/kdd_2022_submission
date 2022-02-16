import torch 
import torch.nn.functional as F 

def multi_task_loss_v2(preds, targets, device):
    # preds: [bsz, # outer class + # inner class]
    # targets: [bsz]
    if (targets == 0).sum():
        loss_outer_normal = [F.binary_cross_entropy_with_logits(preds[targets == 0][:, i], torch.zeros(preds[targets == 0].shape[0]).to(device)) for i in range(3)]
        loss_outer_normal = torch.stack(loss_outer_normal, dim=0).mean(0)
    else:
        loss_outer_normal = torch.tensor(0).to(device)
    
    if (targets == 1).sum():
        loss_outer_damage =  F.binary_cross_entropy_with_logits(preds[targets == 1][:, 0], torch.ones(preds[targets == 1].shape[0]).to(device))
    else:
        loss_outer_damage = torch.tensor(0).to(device) 
    if (targets == 2).sum():
        loss_outer_dirt =  F.binary_cross_entropy_with_logits(preds[targets == 2][:, 1], torch.ones(preds[targets == 2].shape[0]).to(device))
    else:
        loss_outer_dirt = torch.tensor(0).to(device)
    if (targets == 3).sum():
        loss_outer_wash =  F.binary_cross_entropy_with_logits(preds[targets == 3][:, 2], torch.ones(preds[targets == 3].shape[0]).to(device))
    else:
        loss_outer_wash = torch.tensor(0).to(device)
  
    inner_idx = torch.isin(targets, torch.tensor([0,1,2,3]).to(device), invert=True)

    if len(inner_idx):
        loss_inner = F.cross_entropy(preds[inner_idx][:, 3:], targets[inner_idx]-4)  
    else:
        loss_inner = torch.tensor(0).to(device) 

    loss = loss_outer_normal + loss_outer_damage + loss_outer_dirt + loss_outer_wash + loss_inner
    
    loss_dict = {'outer_normal': loss_outer_normal.item(), 
                    'outer_damage': loss_outer_damage.item(), 
                    'outer_dirt': loss_outer_dirt.item(),
                    'outer_wash': loss_outer_wash.item(),
                    'inner': loss_inner.item()} 
   
    return loss, loss_dict

def multi_task_loss(preds, targets, device):
    # preds: [bsz, # outer class + # inner class]
    # targets: [bsz]
    if (targets == 0).sum():
        loss_outer_normal = [F.binary_cross_entropy_with_logits(preds[i][targets == 0], torch.zeros_like(preds[i][targets == 0])) for i in range(3)]
        loss_outer_normal = torch.stack(loss_outer_normal, dim=0).mean(0)
    else:
        loss_outer_normal = torch.tensor(0).to(device)
    
    if (targets == 1).sum():
        loss_outer_damage =  F.binary_cross_entropy_with_logits(preds[0][targets == 1], torch.ones_like(preds[0][targets == 1]))
    else:
        loss_outer_damage = torch.tensor(0).to(device) 
    if (targets == 2).sum():
        loss_outer_dirt =  F.binary_cross_entropy_with_logits(preds[1][targets == 2], torch.ones_like(preds[1][targets == 2]))
    else:
        loss_outer_dirt = torch.tensor(0).to(device)
    if (targets == 3).sum():
        loss_outer_wash =  F.binary_cross_entropy_with_logits(preds[2][targets == 3], torch.ones_like(preds[2][targets == 3]))
    else:
        loss_outer_wash = torch.tensor(0).to(device)
  
    inner_idx = torch.isin(targets, torch.tensor([0,1,2,3]).to(device), invert=True)

    if len(inner_idx):
        loss_inner = F.cross_entropy(preds[3][inner_idx], targets[inner_idx]-4)  
    else:
        loss_inner = torch.tensor(0).to(device) 

    loss = loss_outer_normal + loss_outer_damage + loss_outer_dirt + loss_outer_wash + loss_inner
    
    loss_dict = {'outer_normal': loss_outer_normal.item(), 
                    'outer_damage': loss_outer_damage.item(), 
                    'outer_dirt': loss_outer_dirt.item(),
                    'outer_wash': loss_outer_wash.item(),
                    'inner': loss_inner.item()} 
   
    return loss, loss_dict


def multi_task_loss_v3(preds, targets, device):
    # preds: [bsz, # outer class + # inner class]
    # targets: [bsz]
 
    if (targets == 0).sum():
        loss_outer_normal = [F.cross_entropy(preds[i][targets == 0], torch.zeros(preds[i][targets == 0].shape[0]).long().to(device)) for i in range(3)]
        loss_outer_normal = torch.stack(loss_outer_normal, dim=0).mean(0)
    else:
        loss_outer_normal = torch.tensor(0).to(device)
    
    if (targets == 1).sum():
        loss_outer_damage =  F.cross_entropy(preds[0][targets == 1], torch.ones(preds[0][targets == 1].shape[0]).long().to(device))
    else:
        loss_outer_damage = torch.tensor(0).to(device) 
    if (targets == 2).sum():
        loss_outer_dirt =  F.cross_entropy(preds[1][targets == 2], torch.ones(preds[1][targets == 2].shape[0]).long().to(device))
    else:
        loss_outer_dirt = torch.tensor(0).to(device)
    if (targets == 3).sum():
        loss_outer_wash =  F.cross_entropy(preds[2][targets == 3], torch.ones(preds[2][targets == 3].shape[0]).long().to(device))
    else:
        loss_outer_wash = torch.tensor(0).to(device)
  
    inner_idx = torch.isin(targets, torch.tensor([0,1,2,3]).to(device), invert=True)

    if len(inner_idx):
        loss_inner = F.cross_entropy(preds[3][inner_idx], targets[inner_idx]-4)  
    else:
        loss_inner = torch.tensor(0).to(device) 

    loss = loss_outer_normal + loss_outer_damage + loss_outer_dirt + loss_outer_wash + loss_inner
    
    loss_dict = {'outer_normal': loss_outer_normal.item(), 
                    'outer_damage': loss_outer_damage.item(), 
                    'outer_dirt': loss_outer_dirt.item(),
                    'outer_wash': loss_outer_wash.item(),
                    'inner': loss_inner.item()} 
   
    return loss, loss_dict
