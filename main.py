import os
import json
from collections import defaultdict

import torch
import random

import numpy as np
from torchvision.models import resnet18

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

from torch import nn
from client import Client
from datasets.femnist import Femnist
from server import Server
from utils.args import get_parser
from datasets.idda import IDDADataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from models.cnn import conv_network
from models.margnet import marg_network
from models.FedGen import adv_network
from models.FedRCN import RCN_network
from models.FedADG import c_network
from models.FedSR import network
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics

from wandb_logger import wandb_logger

import matplotlib.pyplot as plt
import torchvision as torchvision

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

    if args.model == 'margnet' or args.model == 'margnetRnd': 
        model = marg_network(args)
        return model

    if args.model == 'FedGen': 
        model = adv_network()
        return model

    if args.model == 'FedRCN': 
        model = RCN_network()
        return model

    if args.model == 'FedADG': 
        model = c_network()
        return model
    
    if args.model == 'FedSR': 
        model = network()
        return model

    raise NotImplementedError


def get_transforms(args, train, test):
    """
    Tranforms both the training and testing data depending on our models and purposes
    """
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

    elif args.model == 'cnn' or args.model == 'resnet18' or args.model == 'margnet' or\
         args.model == 'FedGen' or args.model == 'FedRCN' or args.model == 'margnetRnd'or\
         args.model == 'FedADG' or args.model == 'FedSR':
         
      if args.augment=='off':
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])

      else: 
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
            nptr.Rotate(train)
        ])
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
            nptr.Rotate(test)
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
        cdata = json.load(inf)
        data.update(cdata['user_data'])

    return data


def read_femnist_data(train_data_dir, test_data_dir):
    return read_femnist_dir(train_data_dir), read_femnist_dir(test_data_dir)


def get_datasets(args):
    """
    Method that creates the datasets with the transforms depending on the args
    """

    if args.dataset == 'idda':
        train_datasets = []
        train_transforms, test_transforms = get_transforms(args)
        root = 'data/idda'
        with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)
        for client_id in all_data.keys():
            train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                              client_name=client_id))
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

    elif args.dataset == 'femnist':
        niid = args.niid
        train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
        test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')
        train_data, test_data = read_femnist_data(train_data_dir, test_data_dir)

        train_datasets, test_datasets = [], []

        if args.augment == 'off':
          train_transforms, test_transforms = get_transforms(args, None, None)
          for user, data in train_data.items():
              train_datasets.append(Femnist(data, train_transforms, user))
          for user, data in test_data.items():
              test_datasets.append(Femnist(data, test_transforms, user))
        else: 
          angles = [0, 15, 30, 45, 60, 75]
          angles.remove(args.angle_out)

          _, test_transforms = get_transforms(args, 0, args.angle_out)

          for user, data in train_data.items():
            rotation_train = random.choice(angles)
            train_transforms, _ = get_transforms(args, rotation_train, args.angle_out)
            train_datasets.append(Femnist(data, train_transforms, user))

          for user, data in test_data.items():
            # rotation_train = random.choice(angles)
            # _, test_transforms = get_transforms(args, 0, rotation_train)
            test_datasets.append(Femnist(data, test_transforms, user))

          train_datasets = random.sample(train_datasets, 1000)
          test_datasets = random.sample(test_datasets, 1000)
    else:
        raise NotImplementedError

    return train_datasets, test_datasets


def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'resnet18' or args.model == 'cnn' or args.model == 'margnet' or\
         args.model == 'FedGen' or args.model == 'FedRCN' or args.model == 'margnetRnd' or\
         args.model == 'FedADG' or args.model == 'FedSR':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics


def gen_clients(args, train_datasets, test_datasets, model):
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model, test_client=i == 1))
    return clients[0], clients[1]


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    # Initializing the wandb logger
    if args.wandb == 'on': 
      log = wandb_logger(args)

    print(f'Initializing model...')
    model = model_init(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Done.')

    print('Generating datasets...')
    train_datasets, test_datasets = get_datasets(args)
    print('Done.')

    metrics = set_metrics(args)
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
    
    server = Server(args, train_clients, test_clients, model, metrics)
    server.train()


if __name__ == '__main__':
    main()
