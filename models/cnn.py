import torch.nn as nn
import torch
import torch.distributions as distributions

### Our main convolutional network

class conv_network(torch.nn.Module):

    def __init__(self):

        # Various initializations
        super(conv_network, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Classification layers
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), device=device)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), device=device)
        self.pool = nn.MaxPool2d((2, 2))
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Fully connected part
        self.fc_1 = nn.Linear(in_features=4*4*64, out_features=2048, device=device)
        self.fc_2 = nn.Linear(in_features=2048, out_features=62, device=device)
        self.z_dim = 1024
      
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

    # Network until the last fully connected layer
    def net(self, x):
      x = self.conv_1(x)
      x = self.pool(x)
      x = self.conv_2(self.act(x))
      x = self.pool(x)
      
      x = torch.flatten(x, 1)
      x = self.fc_1(self.act(x))
      return x

    # Featurizer part
    def featurize(self, z_params, num_samples=1): 

        z_mu = z_params[:,:self.z_dim]
        z_sigma = nn.functional.softplus(z_params[:,self.z_dim:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu,z_sigma),1)
        z = z_dist.rsample([num_samples]).view([-1,self.z_dim])

        return z, (z_mu,z_sigma)
