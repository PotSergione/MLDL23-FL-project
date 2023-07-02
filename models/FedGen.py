import torch.nn as nn
import torch
from torch.nn.parallel.distributed import _tree_unflatten_with_rref


## Adversarial network called FedGEN
class adv_network(torch.nn.Module):

    def __init__(self):

        # Various initializations
        super(adv_network, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 62
        self.features_dim = 2048

        # Classification layers
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), device=self.device)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), device=self.device)
        self.pool = nn.MaxPool2d((2, 2))
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Fully connected part
        self.fc_1 = nn.Linear(in_features=4*4*64, out_features=self.features_dim, device=self.device)
        self.fc_2 = nn.Linear(in_features=self.features_dim, out_features=self.num_classes, device=self.device)

        # Generator regularizer
        self.alpha = 0.99
        self.avg = torch.zeros((self.num_classes, self.features_dim), device=self.device, 
                              requires_grad=False)
        self.cov_mat = torch.zeros((self.num_classes, self.features_dim, self.features_dim), 
                              device=self.device, requires_grad=False)
   
    # Training part
    def forward(self, x):

      x = self.conv_1(x)
      x = self.pool(x)
      x = self.conv_2(self.act(x))
      x = self.pool(x)
      
      x = torch.flatten(x, 1)
      x = self.fc_1(self.act(x))
      x = self.fc_2(self.act(x))

      return x

    # Features extractor part
    def features_extractor(self, x):

      x = self.conv_1(x)
      x = self.pool(x)
      x = self.conv_2(self.act(x))
      x = self.pool(x)
      
      x = torch.flatten(x, 1)
      x = self.fc_1(self.act(x))

      return x
    
    # Training part
    def get_reg(self, x, y):
      
      feat = self.features_extractor(x)
      generated = torch.zeros(y.size(0), self.features_dim, device=self.device, requires_grad=False)
      for i in range(y.size(0)): 
        self.avg[y[i]] = self.alpha*self.avg[y[i]] + (1-self.alpha)*feat[i].detach()
        self.cov_mat[y[i], :, :] = self.alpha*self.cov_mat[y[i], :, :] + \
        (1-self.alpha)*torch.matmul((feat[i].detach()-self.avg[y[i]]),(feat[i].detach()-self.avg[y[i]]).transpose(-1,0))
        generated[i] = torch.normal(self.avg[y[i]].transpose(-1,0), self.cov_mat[y[i], :, :])[0]
      
      return feat.norm(dim=1).mean() + (generated-feat).norm(dim=1).mean()
      