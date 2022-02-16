import argparse

from utils.training import train
from utils.data import create_dataloader, name_to_class
from model.set_model import set_model

parser = argparse.ArgumentParser()

parser.add_argument('--arch', default='pmg', choices=['resnet', 'pmg'])
parser.add_argument('--method', default='uni', choices=['uni', 'multi', 'baseline'])

parser.add_argument('--dataset', default='sofar_v3')
parser.add_argument('--data_root_path', default='/home/jovyan/shared/users/tigger/dataset/', type=str)

parser.add_argument('--ce_label', action='store_true', default=True)  
parser.add_argument('--train_class', default='outer_normal,outer_damage,outer_dirt,outer_wash,inner_wash,inner_dashboard,inner_cupholder,inner_glovebox,inner_washer_fluid,inner_rear_seat,inner_sheet_dirt', type=str)
parser.add_argument('--test_class', default='outer_normal,outer_damage,outer_dirt,outer_wash,inner_wash,inner_dashboard,inner_cupholder,inner_glovebox,inner_washer_fluid,inner_rear_seat,inner_sheet_dirt', type=str)

parser.add_argument('--batch_size', default=16, type=int)                          
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--label_smooth', default=0, type=float)
parser.add_argument('--lr', default=2e-3, type=float)
parser.add_argument('--lr_decay', default='cosine', choices=['none', 'cosine', 'plateau'])
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)

parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--exp_name', default='', type=str)

args = parser.parse_args()

print('train class:', args.train_class)
print('test class:', args.test_class)
args.train_class_name = args.train_class 
args.test_class_name = args.test_class 
args.train_class = [name_to_class[item] for item in args.train_class.split(',')]
args.test_class = [name_to_class[item] for item in args.test_class.split(',')]

if args.method == 'baseline':
    assert len(args.train_class) == 2
else:
    assert len(args.train_class) == 11

if __name__  == '__main__':
    ## model ##
    model = set_model(args)
    ###########

    ## data loader ##
    train_loader, test_loader = create_dataloader(args)
    ###########
    
    ## train ## 
    train(model, train_loader, test_loader, args)
    ###########