import torch 
import torch.nn as nn 
import torchvision

def set_model(args):
    if args.arch == 'resnet':
        model = torchvision.models.resnet50(pretrained=True)
        in_feats = model.fc.in_features

        if args.method == 'multi':
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.fc = [nn.Linear(in_feats, 1).to(device), nn.Linear(in_feats, 1).to(device), 
                        nn.Linear(in_feats, 1).to(device), nn.Linear(in_feats, 7).to(device)]
        
        else:
            model.fc = nn.Sequential(
                nn.Linear(
                    in_feats,
                    len(args.train_class) 
            ))
    
    elif args.arch == 'pmg':
        from model.resnet import resnet50
        from model.pmg import PMG, PMG_Multi     
        model = resnet50(pretrained=True)
        
        if args.method == 'multi':
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = PMG_Multi(model, 512, device)
        else:
            model = PMG(model, feature_size = 512, num_classes = len(args.train_class))
    
    return model 