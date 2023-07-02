import torch.nn as nn
import torch
import datasets.np_transforms as nptr

## Network implemented on margnet network base
## RCN stands for: Random Compositions and Noise

class RCN_network(torch.nn.Module):

    def __init__(self):

        # Various initializations
        super(RCN_network, self).__init__()
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

        # Tranformations
        self.transf = nptr.Compose([nptr.RotateTransform([0,15,30,45,60]), nptr.RandomErasing(mean=[0,0,0])])

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


    # Features extraction part
    def features_extractor(self, x):

      x = self.conv_1(x)
      x = self.pool(x)
      x = self.conv_2(self.act(x))
      x = self.pool(x)
      
      x = torch.flatten(x, 1)
      x = self.fc_1(self.act(x))

      return x
    
    # Noise generator part
    def gen_noise(self, x, lam): 
      """
      Generates features from an augmented sample
      """
      y = self.transf(x)
      feat = self.features_extractor(y)
      varm = lam * torch.ones((feat.shape[0], feat.shape[1]), device = self.device) 
      noise_feat = torch.distributions.Independent(torch.distributions.normal.Normal(feat,varm),1)

      return noise_feat.rsample([1]).view([-1,feat.shape[1]])
    
    # Contrastive loss criterion
    def criterion(self, x1, x2, label, margin: float = 1.0):
      """
      Computes the contrastive loss
      """
      dist = nn.functional.pairwise_distance(x1, x2)
      loss = (1 - label) * torch.pow(dist, 2) \
          + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
      loss = torch.mean(loss)

      return loss

    # Computes the regularizing loss
    def regularizing_loss(self, x): 
      """
      Regularizes the two feature vectors
      """
      features_gen = self.gen_noise(x, lam = 0.1)
      features_orig = self.features_extractor(x)
      label = -torch.ones(features_gen.shape[0], device=self.device)

      loss = self.criterion(features_gen, features_orig, label)

      return  loss
