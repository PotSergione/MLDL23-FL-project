import torch.nn as nn
import torch
from torch.nn.modules.loss import L1Loss


## Adversarial network called FedADG

class c_network(torch.nn.Module):

    def __init__(self):

        # Various initializations
        super(c_network, self).__init__()
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

        # Generator part
        self.gc1 = nn.Linear(in_features=self.num_classes + self.features_dim, 
                              out_features=self.num_classes + self.features_dim,
                              device=self.device)
        self.gc2 = nn.Linear(in_features=self.num_classes + self.features_dim, 
                              out_features=self.features_dim, device=self.device)
        self.act_gen = nn.LeakyReLU()
        self.criterion_gen = nn.MSELoss()

        # Discriminator part
        self.compressed_dim = 100
        self.random_projector = torch.rand(self.features_dim+self.num_classes, self.compressed_dim,
                                  requires_grad=False, device=self.device)

        self.dc1 = nn.Linear(in_features=self.compressed_dim, out_features=self.features_dim, 
                            device=self.device)

        self.dc2 = nn.Linear(in_features=self.features_dim, out_features=1, device=self.device)
        self.criterion_disc = nn.BCELoss()
        self.prob = nn.Sigmoid()
   
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
        
    def generator(self, y):

        # Constructing the input
        x = torch.randn(y.size(0), self.features_dim+self.num_classes, device=self.device, 
                        requires_grad=False)
        encoded = nn.functional.one_hot(y)
        x[:, :encoded.size(1)] = encoded

        # Obtaining the output
        x = self.gc1(x)

        return self.act_gen(self.gc2(self.act(x)))
    
    def train_gen(self, x, y, features):

        outs = self.generator(y)

        return self.criterion_gen(outs, features)
  
    def discriminator(self, x, y): 

      xx = torch.zeros(y.size(0), self.num_classes+self.features_dim, 
                      requires_grad=False, device=self.device)
      encoded = nn.functional.one_hot(y.detach())
      xx[:, :encoded.size(1)] = encoded
      xx[:, self.num_classes:] = x

      # Obtaining the output
      xx = torch.matmul(xx, self.random_projector)
      xx = self.dc1(xx)

      return self.prob(self.dc2(self.act(xx)))

    def train_disc(self, x, y, real): 

      # real = 1 for true samples, real = 0 for generated samples
      label = torch.full((y.size(0),1), real, dtype=torch.float, device=self.device, requires_grad=False)
      outputs = self.discriminator(x, y)
      loss = self.criterion_disc(outputs, label)

      return loss
