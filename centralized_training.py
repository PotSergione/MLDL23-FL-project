import os
import json
from collections import defaultdict

import torch
import random
import torchvision

import numpy as np
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import tqdm

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

from torch import nn
from datasets.femnist import Femnist
from utils.args import get_parser
from datasets.idda import IDDADataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from models.cnn import conv_network
from wandb_logger import wandb_logger
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset_num_classes(dataset):
    if dataset == 'idda':
        return 16
    if dataset == 'femnist':
        return 62
    raise NotImplementedError


def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'cnn':
        model = conv_network()
        return model
    raise NotImplementedError


def get_transforms(args):
    if args.model == 'deeplabv3_mobilenetv2':
        train_transforms = sstr.Compose([
            sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.model == 'cnn' or args.model == 'resnet18' and args.augment=='on':
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
            nptr.RotateTransform([0, 15, 30, 45, 60, 75])
        ])
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
            nptr.RotateTransform([0, 15, 30, 45, 60, 75])
        ])
    elif args.model == 'cnn' or args.model == 'resnet18' and args.augment=='off':
      train_transforms = nptr.Compose([
          nptr.ToTensor(),
          nptr.Normalize((0.5,), (0.5,)),
      ])
      test_transforms = nptr.Compose([
          nptr.ToTensor(),
          nptr.Normalize((0.5,), (0.5,)),
      ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms


def read_femnist_dir(data_dir):
    data = defaultdict(lambda: {})
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
          data.update(json.load(inf)['user_data'])
    return data


def read_femnist_data(train_data_dir, test_data_dir):
    return read_femnist_dir(train_data_dir), read_femnist_dir(test_data_dir)


def get_datasets(args):

    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'femnist':
      niid = args.niid
      train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
      test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')
      train_data, test_data = read_femnist_data(train_data_dir, test_data_dir)

      
      train_transforms, test_transforms = get_transforms(args)

      data_train = {'0':{'x':[], 'y':[]}}
      i = 1
      for user, data in train_data.items(): 
        data_train['0']['x'].extend(data['x']) 
        data_train['0']['y'].extend(data['y']) 
        i += 1
        if i == 1000 and args.augment == 'on':
          break

      data_test = {'0':{'x':[], 'y':[]}}
      i = 1
      for user, data in test_data.items(): 
        data_test['0']['x'].extend(data['x']) 
        data_test['0']['y'].extend(data['y']) 
        i += 1
        if i == 1000 and args.augment == 'on': 
          break

      user = '0'
      data_train=data_train['0']
      train_datasets = Femnist(data_train, train_transforms, user)
      
      data_test=data_test['0']
      test_datasets = Femnist(data_test, test_transforms, user)
    else:
        raise "Available only for Femnist"

    return train_datasets, test_datasets

def get_loss_function():
    """Returns, in our case is the CrossEntropy"""
    return torch.nn.CrossEntropyLoss()

def train(net, data_loader, optimizer, loss_function, device=device):
    """Trains the net using the samples of the trainloader"""
    samples = 0  
    cumulative_loss = 0
    cumulative_accuracy = 0

    net.train()
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(data_loader)):

      # This batch sends the data to the device (in my case the GPU)
      inputs = inputs.to(device)
      targets = targets.to(device)

      # Here we compute the loss and then apply the optimizer
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = loss_function(outputs, targets)
      loss.backward()
      optimizer.step()

      # Statistics to display at training time
      samples += targets.size(0)
      cumulative_loss += loss.item()
      predicted = outputs.max(dim=1)[1] 
      cumulative_accuracy += torch.sum((predicted == targets))
        
    
    return cumulative_loss / samples, (cumulative_accuracy / samples)


def test(net, data_loader, loss_function, device=device):
    """Tests the net using samples of the testloader"""

    samples = 0  
    cumulative_loss = 0
    cumulative_accuracy = 0

    net.eval()
    with torch.no_grad(): 

      for batch_idx, (inputs, targets) in enumerate(data_loader):

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = net(inputs)
        outputs = outputs.to(device)
        predicted = outputs.max(dim=1)[1]

        cumulative_loss += loss_function(outputs, targets)
        cumulative_accuracy += torch.sum(predicted == targets)
        samples += targets.size(0)

    return cumulative_loss / samples, (cumulative_accuracy / samples)


def main():

    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    learning_rate = args.lr 
    epochs = args.num_epochs
    weight_decay = args.wd 
    momentum = args.m
    BATCH_SIZE = args.bs
    set_seed(args.seed)

    # Initializing the wandb logger
    if args.wandb == 'on': 
      logger = wandb_logger(args)

    print(f'Initializing model...')
    model = model_init(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Done.')

    print('Generating datasets...')
    train_dataset, test_dataset = get_datasets(args)
    print('Done.')

    print('Generating the dataloaders')
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True)
    print('Done.')

    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.m, weight_decay=args.wd)
    loss_function = get_loss_function()

    for e in range(1, epochs):
      train_loss, train_accuracy = train(model, train_loader, optimizer, loss_function)
      test_loss, test_accuracy = test(model, test_loader, loss_function)

      if args.print_train_interval % e == 0: 
        print('Epoch: {:d}'.format(e + 1))
        print('\t{}  {:.5f}, {} {:.2f}'.format('Training loss', train_loss, 'Training accuracy', train_accuracy))
      elif args.print_test_interval % e == 0: 
        print('\t{}  {:.5f}, {} {:.2f}'.format('Test loss', test_loss, 'Test accuracy', test_accuracy))
        print('---------------------------------------------------------')  
        
        if args.wandb == 'on' and args.test_interval%e == 0: 
          wandb.log({"accuracy":test_accuracy})


if __name__ == '__main__':
    main()
