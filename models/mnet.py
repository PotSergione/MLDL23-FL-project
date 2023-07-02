import torch.nn as nn
import torch
import datasets.np_transforms as nptr
import torchvision.transforms as tvt


class marg_network(torch.nn.Module):
    def __init__(self, args):
        super(marg_network, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 62
        self.features_dim = 2048
        self.augmented = args.augment
        self.marg = args.model

        # classification layers
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), device=self.device)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), device=self.device)
        self.pool = nn.MaxPool2d((2, 2))
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.fc_1 = nn.Linear(in_features=4*4*64, out_features=self.features_dim, device=self.device)
        self.fc_2 = nn.Linear(in_features=self.features_dim, out_features=self.num_classes, device=self.device)

        # tranformations
        if self.augmented == 'on':
            rotations = [5,20,35]
        else: rotations = [15,30,45,60,75]
        self.transf = [nptr.RotateTransform(rotations), 
                       tvt.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),\
                       tvt.GaussainBlur(kernel_size=3,sigma=0.5),\
                       nptr.RandomErasing(probability=1, sl=0.05,sh=0.1, r1=0.5, 
                       mean = [0,0,0])]

    def forward(self, x):
      x = self.conv_1(x)
      x = self.pool(x)
      x = self.conv_2(self.act(x))
      x = self.pool(x)
      
      x = torch.flatten(x, 1)
      x = self.fc_1(self.act(x))
      x = self.fc_2(self.act(x))

      return x

    def features_extractor(self, x):
      x = self.conv_1(x)
      x = self.pool(x)
      x = self.conv_2(self.act(x))
      x = self.pool(x)
      
      x = torch.flatten(x, 1)
      x = self.fc_1(self.act(x))
      return x
    
    def generator(self, x): 
      """
      Generates features from an augmented sample
      """
      if self.marg == 'margnet':
	      idx = torch.randint(0, 4, (1,1))
      else: 
        idx = torch.tensor(0)

      y = self.transf[idx.item()](x)
      feat = self.features_extractor(y)
      return feat
    
    def criterion(self, x1, x2, label, margin: float = 1.0):
      """
      computes contrastive loss
      """
      dist = nn.functional.pairwise_distance(x1, x2)
      loss = (1 - label) * torch.pow(dist, 2) \
          + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
      loss = torch.mean(loss)
      return loss

    def regularizing_loss(self, x): 
      """
      regularizes the two feature vectors
      """
      features_gen = self.generator(x)
      features_orig = self.features_extractor(x)
      label = -torch.ones(features_gen.shape[0], device=self.device)

      loss = self.criterion(features_gen, features_orig, label)

      return  loss