import os
import numpy as np
from tqdm import tqdm 
from itertools import chain
from sklearn.metrics import *

import wandb 

import torch 
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from utils.data import jigsaw_generator
from utils.losses import * 
from utils.metric import calculate_metrics

def train(model, train_loader, test_loader, args):
    # model 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # optimizer 
    optimizer, scheduler = load_optimizer(model, args.arch, args.method, args.lr, args.lr_decay, args.weight_decay, args.epochs, args.momentum)

    # for logging
    wandb.init(project="KDD22_car_state_cls", name= args.arch + '_' + args.method + '_' + args.exp_name)
    model_save_base = os.path.join('./artifacts', args.arch + '_' + args.method, args.exp_name)    
    
    if not os.path.exists(model_save_base):
        os.makedirs(model_save_base)
    
    # iteration
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        print('> Epoch: ', epoch)
        # train epoch
        train_acc, train_prec, train_rec, train_f1, train_loss, train_loss_dict = train_epoch(train_loader, model, optimizer, args.label_smooth, 
                                                                                        args.method, args.arch, device)
        # eval 
        test_acc, test_prec, test_rec, test_f1, test_loss, test_loss_dict  = test(test_loader, model, args.method, args.arch, device)
        
        pbar.set_description('[Train] Acc: {:.3f}, precision: {:.3f} || [TEST] Acc: {:.3f}, precision {:.3f}'.format(
                                        train_acc, train_prec, test_acc, test_prec))
        wandb.log({'loss/train': train_loss, 'loss/test': test_loss, 
                'acc/train': train_acc, 'acc/test': test_acc,
                'prec/train': train_prec, 'prec/test': test_prec,
                'rec/train': train_rec, 'rec/test': test_rec,
                'f1/train': train_f1, 'f1/test': test_f1,
                'lr':optimizer.param_groups[0]['lr']
        })

        # if args.method == 'multi':
        #     wandb.log({'loss_log/train_outer_normal': train_loss_dict['outer_normal'],
        #                     'loss_log/train_outer_damage': train_loss_dict['outer_damage'],
        #                     'loss_log/train_outer_dirt': train_loss_dict['outer_dirt'],
        #                     'loss_log/train_outer_wash': train_loss_dict['outer_wash'],
        #                     'loss_log/train_inner': train_loss_dict['inner'],
        #             })
        #     wandb.log({'loss_log/test_outer_normal': test_loss_dict['outer_normal'],
        #                     'loss_log/test_outer_damage': test_loss_dict['outer_damage'],
        #                     'loss_log/test_outer_dirt': test_loss_dict['outer_dirt'],
        #                     'loss_log/test_outer_wash': test_loss_dict['outer_wash'],
        #                     'loss_log/test_inner': test_loss_dict['inner'],
        #             })

        
        if args.lr_decay == 'cosine':
            scheduler.step()
        elif args.lr_decay == 'plateau':
            scheduler.step(test_loss)

    model_save_path = os.path.join(model_save_base, 'last.pth')
    torch.save(model.state_dict(), model_save_path)
    print('trained model is saved!')

    print('[Results] Acc || Prec. || Rec. || F1')
    print('{:.3f} {:.3f} {:.3f} {:.3f}'.format(test_acc, test_prec, test_rec, test_f1))

def train_epoch(dataloader, net, optimizer, label_smooth, method, arch, device):
    # ce_loss = torch.nn.CrossEntropyLoss()
    ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smooth)
    loss_dict = {}
    train_losses = list();train_preds = list();train_trues = list()

    net.train()
    for idx, (img, label) in enumerate(dataloader):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        if arch == 'pmg':
            input1 = jigsaw_generator(img, 8)
            input2 = jigsaw_generator(img, 4)
            input3 = jigsaw_generator(img, 2)
            
            optimizer.zero_grad()
            out1, _, _, _ = net._forward(input1)
            if 'multi' not in method:
                loss1 = ce_loss(out1, label) * 1
            elif method == 'multi_v2':
                loss1, _ = multi_task_loss_v2(out1, label, device) 
            else:
                loss1, _ = multi_task_loss(out1, label, device) 

            loss1.backward()
            optimizer.step()

            optimizer.zero_grad()
            _, out2, _, _ = net._forward(input2)
            if 'multi' not in method:
                loss2 = ce_loss(out2, label) * 1
            elif method == 'multi_v2':
                loss2, _ = multi_task_loss_v2(out2, label, device) 
            else:
                loss2, _ = multi_task_loss(out2, label, device)
            
            loss2.backward()
            optimizer.step()

            optimizer.zero_grad()
            _, _, out3, _ = net._forward(input3)
            if 'multi' not in method:
                loss3 = ce_loss(out3, label) * 1
            elif method == 'multi_v2':
                loss3, _ = multi_task_loss_v2(out3, label, device) 
            else:
                loss3, _ = multi_task_loss(out3, label, device) 

            loss3.backward()
            optimizer.step()

            optimizer.zero_grad()
            _, _, _, out4 = net._forward(img)
            if 'multi' not in method:
                loss4 = ce_loss(out4, label)
            elif method == 'multi_v2':
                loss4, _ = multi_task_loss_v2(out4, label, device) 
            else:
                loss4, loss_dict = multi_task_loss(out4, label, device) 
            loss4 *= 2 
            
            loss4.backward()
            optimizer.step()
            
            loss = loss1 + loss2 + loss3 + loss4 
          
            if 'multi' not in method:
                _, pred = torch.max(out4, 1)
                train_preds.extend(pred.view(-1).cpu().detach().numpy().tolist())
            elif method == 'multi_v2':
                out4[:, 0] = torch.sigmoid(out4[: ,0])
                out4[:, 1] = torch.sigmoid(out4[: ,1])
                out4[:, 2] = torch.sigmoid(out4[: ,2])
                out4[:, 3:] = F.softmax(out4[: ,3:], dim=1)                

                max_vals, max_idxs = torch.max(out4,1)
                pred = []
                for max_val, max_idx in zip(max_vals, max_idxs):
                    if max_val < 0.5:
                        pred.append(0)
                    else:
                        pred.append(max_idx.item() + 1)
                train_preds.extend(pred)
            else:
                # out_damage, out_dirt, out_wash, out_inner = out4 
                out4 = torch.cat(out4, dim=1)
                
                max_vals, max_idxs = torch.max(out4,1)
                pred = []
                for max_idx in max_idxs:
                    if max_idx in [0,2,4]:
                        pred.append(0)
                    elif max_idx == 1:
                        pred.append(1)
                    elif max_idx == 3:
                        pred.append(2)
                    elif max_idx == 5:
                        pred.append(3)
                    else:
                        pred.append(max_idx.item() - 2)
                train_preds.extend(pred)
            
            train_losses.append(loss.item())
            train_trues.extend(label.view(-1).cpu().numpy().tolist())

           
        else:
            out = net(img)

            if 'multi' not in method:
                _, pred = torch.max(out, 1)
                
                loss = ce_loss(out, label)
                
                loss.backward()
                optimizer.step()
                
                train_preds.extend(pred.view(-1).cpu().detach().numpy().tolist())
            elif method == 'multi_v2':
                loss, loss_dict = multi_task_loss_v2(out, label, device) 
                
                loss.backward()
                optimizer.step()
                
                out[:, 0] = torch.sigmoid(out[: ,0])
                out[:, 1] = torch.sigmoid(out[: ,1])
                out[:, 2] = torch.sigmoid(out[: ,2])
                out[:, 3:] = F.softmax(out[: ,3:], dim=1)                

                max_vals, max_idxs = torch.max(out,1)
                pred = []
                for max_val, max_idx in zip(max_vals, max_idxs):
                    if max_val < 0.5:
                        pred.append(0)
                    else:
                        pred.append(max_idx.item() + 1)
                train_preds.extend(pred)
            else:
                loss, loss_dict = multi_task_loss(out, label, device) 
                
                loss.backward()
                optimizer.step()
                
                max_vals, max_idxs = torch.max(out4,1)
                pred = []
                for max_idx in max_idxs:
                    if max_idx in [0,2,4]:
                        pred.append(0)
                    elif max_idx == 1:
                        pred.append(1)
                    elif max_idx == 3:
                        pred.append(2)
                    elif max_idx == 5:
                        pred.append(3)
                    else:
                        pred.append(max_idx.item() - 2)
                train_preds.extend(pred)
           
            train_losses.append(loss.item())
            train_trues.extend(label.view(-1).cpu().numpy().tolist())
    
    acc, f1, prec, rec = calculate_metrics(train_trues, train_preds)
    print('trainset result')
    print(confusion_matrix(train_trues, train_preds))

    return acc, prec, rec, f1, np.mean(train_losses), loss_dict

@torch.no_grad()
def test(dataloader, net, method, arch, device):
    ce_loss = torch.nn.CrossEntropyLoss()
    loss_dict = {}
    test_losses = list();test_trues = list()
    test_preds = list();test_preds_ens = list()

    net.eval()
    for idx, (img, label) in enumerate(dataloader):

        img = img.to(device)
        label = label.to(device)
        
        if arch == 'pmg':
            out1, out2, out3, out_concat = net._forward(img)
            
            if 'multi' not in method:
                loss = ce_loss(out_concat, label)
                _, pred = torch.max(out_concat, 1)
                test_preds.extend(pred.view(-1).cpu().detach().numpy().tolist())

            elif method == 'multi_v2':
                loss, loss_dict = multi_task_loss_v2(out_concat, label, device)
                                
                out_concat[:, 0] = torch.sigmoid(out_concat[: ,0])
                out_concat[:, 1] = torch.sigmoid(out_concat[: ,1])
                out_concat[:, 2] = torch.sigmoid(out_concat[: ,2])
                out_concat[:, 3:] = F.softmax(out_concat[: ,3:], dim=1)                

                max_vals, max_idxs = torch.max(out_concat,1)
                pred = []
                for max_val, max_idx in zip(max_vals, max_idxs):
                    if max_val < 0.5:
                        pred.append(0)
                    else:
                        pred.append(max_idx.item() + 1)
                test_preds.extend(pred)            

            else:
                loss, loss_dict = multi_task_loss(out_concat, label, device)
                                
                out_concat = torch.cat(out_concat, dim=1)
                
                max_vals, max_idxs = torch.max(out_concat,1)
                pred = []
                for max_idx in max_idxs:
                    if max_idx in [0,2,4]:
                        pred.append(0)
                    elif max_idx == 1:
                        pred.append(1)
                    elif max_idx == 3:
                        pred.append(2)
                    elif max_idx == 5:
                        pred.append(3)
                    else:
                        pred.append(max_idx.item() - 2)
                test_preds.extend(pred)
            test_losses.append(loss.item())
            test_trues.extend(label.view(-1).cpu().numpy().tolist())
            # test_preds_ens.extend(pred_ens.view(-1).cpu().detach().numpy().tolist())
        else:
            out = net(img)
            
            if 'multi' not in method:
                _, pred = torch.max(out, 1)
                loss = ce_loss(out, label)
                
                test_preds.extend(pred.view(-1).cpu().detach().numpy().tolist())
            elif method == 'multi_v2':
                loss, loss_dict = multi_task_loss_v2(out, label, device)
                                
                out[:, 0] = torch.sigmoid(out[: ,0])
                out[:, 1] = torch.sigmoid(out[: ,1])
                out[:, 2] = torch.sigmoid(out[: ,2])
                out[:, 3:] = F.softmax(out[: ,3:], dim=1)                

                max_vals, max_idxs = torch.max(out,1)
                pred = []
                for max_val, max_idx in zip(max_vals, max_idxs):
                    if max_val < 0.5:
                        pred.append(0)
                    else:
                        pred.append(max_idx.item() + 1)
                test_preds.extend(pred)            

            else:
            
                loss, loss_dict = multi_task_loss(out, label, device) 
                
                max_vals, max_idxs = torch.max(out,1)
                pred = []
                for max_idx in max_idxs:
                    if max_idx in [0,2,4]:
                        pred.append(0)
                    elif max_idx == 1:
                        pred.append(1)
                    elif max_idx == 3:
                        pred.append(2)
                    elif max_idx == 5:
                        pred.append(3)
                    else:
                        pred.append(max_idx.item() - 2)
                test_preds.extend(pred)
                
            test_losses.append(loss.item())
            test_trues.extend(label.view(-1).cpu().numpy().tolist())


    print('testset result')
    acc, f1, prec, rec = calculate_metrics(test_trues, test_preds)
    print(confusion_matrix(test_trues, test_preds))


    return acc, prec, rec, f1, np.mean(test_losses), loss_dict


def load_optimizer(model, arch, method, lr, lr_decay, weight_decay, epochs, momentum=0.9):
    if arch == 'resnet':
        optimizer = torch.optim.SGD(
                model.parameters(),
                lr = lr,
                momentum=momentum,
                weight_decay = weight_decay,
        )

    elif arch == 'pmg':
        if method == 'multi':
            optimizer = torch.optim.SGD([
                    {'params': model.conv_block1.parameters(), 'lr': lr},
                    {'params': model.classifier1.parameters(), 'lr': lr},
                    {'params': list(chain(*[h.parameters() for h in model.heads1])), 'lr': lr},
                    
                    {'params': model.conv_block2.parameters(), 'lr': lr},
                    {'params': model.classifier2.parameters(), 'lr': lr},
                    {'params': list(chain(*[h.parameters() for h in model.heads2])), 'lr': lr},

                    {'params': model.conv_block3.parameters(), 'lr': lr},
                    {'params': model.classifier3.parameters(), 'lr': lr},
                    {'params': list(chain(*[h.parameters() for h in model.heads3])), 'lr': lr},

                    {'params': model.classifier_concat.parameters(), 'lr':lr},
                    {'params': list(chain(*[h.parameters() for h in model.heads_concat])), 'lr': lr},
                    
                    {'params': model.features.parameters(), 'lr': lr/10}
                ],
                momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD([
                {'params': model.conv_block1.parameters(), 'lr': lr},
                {'params': model.classifier1.parameters(), 'lr': lr},
                
                {'params': model.conv_block2.parameters(), 'lr': lr},
                {'params': model.classifier2.parameters(), 'lr': lr},

                {'params': model.conv_block3.parameters(), 'lr': lr},
                {'params': model.classifier3.parameters(), 'lr': lr},

                {'params': model.classifier_concat.parameters(), 'lr':lr},
                
                {'params': model.features.parameters(), 'lr': lr/10}
            ],
            momentum=momentum, weight_decay=weight_decay)
    if lr_decay == 'none':
        scheduler = None 
    elif lr_decay == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)
    elif lr_decay == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:
        NotImplementedError()
    
    return optimizer, scheduler
