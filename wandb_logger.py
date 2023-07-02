import wandb


class wandb_logger(): 

  def __init__(self, args): 

    if args.niid: 
      self.project = 'niid_federated'
    else: 
      self.project ='iid_federated'
    
    if args.client_selection != 'uniform': 
      self.project = self.project + ' '+args.client_selection

    if args.augment == 'on': 
      self.project = self.project + ' ' + 'augmented'
    
    if args.model == 'margnet':
      self.project = self.project + ' ' +'margnet' 

    if args.model == 'margnetRnd':
      self.project = self.project + ' ' +'margnetRnd' 

    if args.model == 'FedRCN':
      self.project = self.project + ' ' +'FedRCN' 

    if args.model == 'FedGen': 
      self.project = self.project + ' '+'FedGen'

    if args.model == 'FedADG':
      self.project = self.project + ' '+'FedADG'

    if args.FedSR == 'on':
      self.project = self.project + ' ' + 'FedSR'
    
    # Starts a new wandb run to track this script
    wandb.init(

        # Set the wandb project where this run will be logged 
        project= self.project,

        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "architecture": args.model,
            "dataset": "FEMNIST",
            "clients_per_round": args.clients_per_round,
            "rounds": args.num_rounds,
            "epochs": args.num_epochs,
            "optimizer": 'SGD',
            "metric": 'Accuracy',
            "weight_decay": args.wd,
            "momentum": args.m, 
            "seed": args.seed
        }
    )
