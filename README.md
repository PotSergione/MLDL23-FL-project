# Towards Real World Federated Learning
### Machine Learning and Deep Learning 2023
#### Politecnico di Torino
Definitve code for the Federated Learning project 2A by Angelini, Battistotti and Gallo

## Setup
#### Environment
If not working on CoLab, install environment with conda (preferred): 
```bash 
conda env create -f mldl23fl.yml
```

#### Datasets
The repository supports experiments on the following dataset:
  **FEMNIST** (Federated Extended MNIST) from LEAF benchmark [1]
   - Task: image classification on 62 classes
   - 3,597 users
   - Instructions for download and preprocessing in ```data/femnist/``` 

## How to run
The experimental data have been obtained by running the commands: 
```bash
./preprocess.sh -s iid --sf 0.33 -k 0 -t --iu 1.0
```
```bash
./preprocess.sh -s niid --sf 0.33 -k 0 -t
```

In doing so we are undersampling the EMNIST data, keeping only a third of its 
original volume to comply with COLAB's RAM constraints.



The ```main.py``` orchestrates federated training. All arguments need to be specified through the ```args``` parameter (options can be found in ```utils/args.py```).
Example of FedAvg experiment:

```bash
python main.py --niid --dataset femnist --model cnn --num_rounds 1000 --num_epochs 5 --clients_per_round 10 --client_selection poc
```



The ```centralized_training.py``` orchestrates centralized training in both the augmented and standard scenarios. 
All arguments need to be specified through the ```args``` parameter. Example:
```bash
python centralized_training.py --dataset femnist --model cnn --num_epochs 5 --lr 0.01
```

### Logging the experiments
We provide a utility to connect to the Wandb server via simple logger. It requires preventive autentication.

## References
[1] Caldas, Sebastian, et al. "Leaf: A benchmark for federated settings." Workshop on Federated Learning for Data Privacy and Confidentiality (2019). 
