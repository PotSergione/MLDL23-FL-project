import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction
import torch.distributions as distributions

class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.batch_size = self.args.bs
        self.num_classes = 62
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        
        # Added device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)

        # Added loss for the POC communication/computation efficient method
        self.loss = 1e15

        # Added for the entropy client selection
        self.entropy = self.dataset.entropy

        # Added for Margnet and FedRCN
        self.alpha = 0.01

        if self.args.model == 'FedADG':
          self.lclass = 0.85
          self.lerr = 0.15 
       
          
    def __str__(self):
        return self.name


    @staticmethod
    def update_metric(metric, outputs, labels):
      """
        This method updates the objects of the class StreamClsMetric to store the 
        results of the training
      """
      _, prediction = outputs.max(dim=1)
      labels = labels.cpu().numpy()
      prediction = prediction.cpu().numpy()
      metric.update(labels, prediction) 


    def _get_outputs(self, images):
      """
        This method gets the outputs from the specified neural network
        in the case of cnn, it returns the result of calling .forward()
      """
      if self.args.model == 'deeplabv3_mobilenetv2':
          return self.model(images)['out']
      if self.args.model == 'resnet18' or self.args.model == 'cnn' or \
        self.args.model == 'margnet' or self.args.model == 'FedGen' or \
        self.args.model == 'FedRCN' or self.args.model == 'margnetRnd'or\
        self.args.model == 'FedADG' or self.args.model == 'FedSR':
          return self.model(images)
      raise NotImplementedError


    def compute_loss(self, x, y, optimizer):
      """
        This method computes the loss differentiating between all the different methods 
        methods. 
        :param x: images
        :param y: labels
        :param optimizer: optimizer used for the generative network training
        :return: the value of the loss
      """ 
      if self.args.model == 'FedSR': 
        return self.model.regressive_loss(x, y)

      elif self.args.model == 'margnet':
        loss_disc = self.model.regularizing_loss(x)
        outputs = self._get_outputs(x)
        return self.criterion(outputs, y) + self.alpha*loss_disc

      elif self.args.model == 'FedGen': 
        outputs = self._get_outputs(x)
        return self.criterion(outputs, y) + self.model.get_reg(x,y)
        
      elif self.args.model == 'FedADG':  
        feat = self.model.features_extractor(x)
        # training the discriminator
        loss_disc_true = self.model.train_disc(feat, y, 1)
        xx = self.model.generator(y)
        loss_disc_false = self.model.train_disc(xx, y, 0)
        # training the generator
        disc_loss = loss_disc_true + loss_disc_false  
        loss = self.lerr*self.model.train_gen(x, y, feat)
        loss = loss + disc_loss

        outputs = self._get_outputs(x)
        closs = self.lclass*self.criterion(outputs, y)

        return closs + loss

      elif self.args.model == 'FedRCN':
        loss_disc = self.model.regularizing_loss(x)
        outputs = self._get_outputs(x)
        return self.criterion(outputs, y) + self.alpha*loss_disc

      else:
        outputs = self._get_outputs(x)
        return self.criterion(outputs, y) 


    def run_epoch(self, cur_epoch, optimizer):
        """
          This method locally trains the model with the dataset of the client. 
          It handles the training at mini-batch level
          :param cur_epoch: current epoch of training
          :param optimizer: optimizer used for the local training
        """
        self.model.train()
        for cur_step, (images, labels) in enumerate(self.train_loader):
            
          images = images.to(self.device)
          labels = labels.to(self.device)  
          
          optimizer.zero_grad()
          loss = self.compute_loss(images, labels, optimizer)
          self.loss += loss.item()
        
          loss.backward()
          optimizer.step()

            
    def train(self):
        """
          This method locally trains the model with the dataset of the client. 
          It handles the training at epochs level
          (by calling the run_epoch method for each local epoch of training)
          :return: length of the local dataset, copy of the model parameters
        """
        self.loss = 0
        for epoch in range(self.args.num_epochs):
            self.run_epoch(epoch, self.optimizer)
        self.loss = self.loss/len(self.train_loader)
        
        return (len(self.dataset), self.model.state_dict())


    def test(self, metric):
        """
          This method tests the model on the local dataset of the client.
          :param metric: StreamMetric object
        """
        with torch.no_grad():
          for i, (images, labels) in enumerate(self.test_loader):
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self._get_outputs(images)

            self.update_metric(metric, outputs, labels)
