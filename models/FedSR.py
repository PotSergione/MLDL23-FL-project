import torch.nn as nn
import torch
import torch.distributions as distributions

### Our main convolutional network

class network(torch.nn.Module):

    def __init__(self):

        # Various initializations
        super(network, self).__init__()
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

        # regularizing portion
        # Both those parameters are taken from the GitHub implemetation of the algorithm
        self.L2R_coeff = 0.01 
        self.CMI_coeff = 0.001 

        self.r_mu = nn.Parameter(torch.zeros(62,self.z_dim, requires_grad=True, device=device), requires_grad=True)
        self.r_sigma = nn.Parameter(torch.ones(62,self.z_dim, requires_grad=True, device=device), requires_grad=True)
        self.C = nn.Parameter(torch.ones([], requires_grad=True, device=device), requires_grad=True)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
      
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
    
    def regressive_loss(self, x, y):

      """
      Computes the regularizer term of the FedSR model
      """

      out = self.net(x)
      z, (z_mu,z_sigma) = self.featurize(out, num_samples=y.size(0))
      
      outputs = self.fc_2(self.act(out))
      loss = self.criterion(outputs, y)

      obj = loss
      regL2R = torch.zeros_like(obj)
      regCMI = torch.zeros_like(obj)

      # Compute the regularizing terms
      if self.L2R_coeff != 0.0:
          regL2R = z.norm(dim=1).mean()
          obj = obj + self.L2R_coeff*regL2R

      if self.CMI_coeff != 0.0:

          # Here we compute the conditional expectation
          r_sigma_softplus = nn.functional.softplus(self.r_sigma)
          r_mu = self.r_mu[y]
          r_sigma = r_sigma_softplus[y]
          z_mu_scaled = z_mu*self.C
          z_sigma_scaled = z_sigma*self.C
          regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                  (z_sigma_scaled**2+(z_mu_scaled-r_mu)**2)/(2*r_sigma**2) - 0.5
          regCMI = regCMI.sum(1).mean()
          obj = obj + self.CMI_coeff*regCMI 

      return obj
