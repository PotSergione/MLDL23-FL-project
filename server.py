import copy
from collections import OrderedDict

import numpy as np
import torch.optim as optim
import torch

import wandb


class Server:

    # Server class initializations
    def __init__(self, args, train_clients, test_clients, model, metrics):

      self.args = args
      self.train_clients = train_clients
      self.test_clients = test_clients
      self.model = model
      self.metrics = metrics
      self.model_params_dict = copy.deepcopy(self.model.state_dict())
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.num_clients = min(self.args.clients_per_round, len(self.train_clients))

      if self.args.client_selection=='poc': 

        hold = [len(i.train_loader) for i in self.train_clients]
        self.prob = [i/sum(hold) for i in hold]

      elif self.args.client_selection == 'smart10': 
        
        data_len = len(self.train_clients) 
        b = np.random.choice(np.linspace(0, data_len-1, data_len,dtype=int), 
                                              int(data_len/10), replace=False)
        
        self.prob = np.ones((data_len, ))*0.5/(data_len-int(data_len/10))
        self.prob[b] = 0.5/int(data_len/10)

      elif self.args.client_selection == 'smart30': 
        
        data_len = len(self.train_clients)
        b = np.random.choice(np.linspace(0, data_len-1, data_len,dtype=int), 
                                              int(data_len/3), replace=False)
        self.prob = np.ones((data_len, ))*0.9999/(data_len-int(data_len/3))
        self.prob[b] = 0.0001/int(data_len/3)

      elif self.args.client_selection == 'entropy': 

        hold = [i.entropy for i in self.train_clients]
        self.prob = [i/sum(hold) for i in hold]
        

    def select_clients(self, ith_round):
      """
        This method selects the clients that will be used to train the net. 
        5 different alternatives are possible, via arg selection: 
        
        uniform: samples the clients uniformly at random each training round
        smart 10/30: imposes on the 10% (resp. 30%) a c.d.f. of 0.5 (resp 0.0001)
        poc: implements the Power of Choice method; if d is not given, it has 
              default value 20
        entropy: samples the clients via convex combination of loss and entropy of each client
        :return: list of clients to use for training/testing
      """

      if self.args.client_selection == 'uniform': 
        return np.random.choice(self.train_clients, self.num_clients, replace=False)

      elif self.args.client_selection == 'smart10' or self.args.client_selection == 'smart30':
        return np.random.choice(self.train_clients, self.num_clients, replace=False,
          p = self.prob)

      elif self.args.client_selection == 'poc': 

        if self.args.d < self.num_clients:  
          raise ValueError("PoC samples cannot be less than clients per round")

        selected = np.random.choice(self.train_clients, size=self.args.d, 
        replace=False, p=self.prob)

        self.loss_by_client = {}
        for c in selected: 
          self.loss_by_client[c] = c.loss

        self.loss_by_client = dict(sorted(self.loss_by_client.items(), 
                    key=lambda item: item[1]))
        selected = list(self.loss_by_client.keys())[-self.num_clients:]
        self.loss_by_client.clear()
        
        return selected
      
      elif self.args.client_selection == 'entropy': 
        
        # sigma in our method
        sigma = 0.99 

        selected = np.random.choice(self.train_clients, size=self.args.d, 
          replace=False, p=self.prob)

        self.loss_by_client = {}
        for c in selected: 
          self.loss_by_client[c] = c.loss*(1-sigma**ith_round) + \
          (sigma**ith_round)*c.entropy

        self.loss_by_client = dict(sorted(self.loss_by_client.items(), 
                    key=lambda item: item[1]))
        selected = list(self.loss_by_client.keys())[-self.num_clients:]
        self.loss_by_client.clear()
        
        return selected          
      raise NotImplementedError


    def train_round(self, clients):
      """
        This method trains the model with the dataset of the clients. 
        It handles the training at single round level
        :param clients: list of all the clients to train
        :return: model updates gathered from the clients, to be aggregated
      """
      updates = []
      for i, c in enumerate(clients):
          c.model.load_state_dict(copy.deepcopy(self.model_params_dict))
          updates.append(c.train())
      return updates


    def FedAvg(self, updates):
      """
        This method handles the FedAvg/FedIR aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
      """
      total_weight = 0.
      base = OrderedDict()
      for (client_samples, client_model) in updates:
          total_weight += client_samples
          for key, value in client_model.items():
              if key in base:
                  base[key] += (client_samples * value.type(torch.FloatTensor))
              else:
                  base[key] = (client_samples * value.type(torch.FloatTensor))

      averaged_soln = copy.deepcopy(self.model.state_dict())
      for key, value in base.items():
          if total_weight != 0:
              averaged_soln[key] = value.to(self.device) / total_weight

      self.model.load_state_dict(averaged_soln)
      self.model_params_dict = copy.deepcopy(self.model.state_dict())


    def train(self):
      """
        This method orchestrates the training, then evals and tests at rounds
        level every n rounds
      """
      for r in range(self.args.num_rounds + 1):
          print('Round {}...'.format(r))
          clients = self.select_clients(r)
          updates = self.train_round(clients)
          print('Finished clients round!')

          print('Aggregating the data...')
          self.FedAvg(updates)
          print('Data aggregated!')

          if r % self.args.eval_interval == 0 and r>0: 
            self.eval_train()
            self.metrics['eval_train'].get_results()

            if r % self.args.print_train_interval == 0 and r > 0:
              print(self.metrics['eval_train'])
              self.metrics['eval_train'].reset()

          if r % self.args.test_interval == 0 and r>0:
            self.test()
            self.metrics['test'].get_results()

            if r % self.args.print_test_interval == 0 and r > 0:
              print(self.metrics['test'])

              if self.args.wandb =='on': 
                wandb.log(self.metrics['test'].results)

            self.metrics['test'].reset()
      torch.save(self.model.state_dict(), self.args.model)


    def eval_train(self):
        """
          This method handles the evaluation on the train clients
        """
        train_metrics = []
        for i, c in enumerate(self.train_clients):
            c.test(self.metrics['eval_train'])
    

    def test(self):
        """
          This method handles the test on the test clients
        """
        test_metrics = []
        for i, c in enumerate(self.test_clients):
            c.test(self.metrics['test'])
