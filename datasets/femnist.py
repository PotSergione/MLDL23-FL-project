import numpy as np
import datasets.np_transforms as tr

from typing import Any
from torch.utils.data import Dataset
import torch

IMAGE_SIZE = 28

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])


class Femnist(Dataset):

    def __init__(self,
                 data: dict,
                 transform: tr.ToTensor,
                 client_name: str):
        super().__init__()
        self.samples = [(image, label) for image, label in zip(data['x'], data['y'])]
        self.transform = transform
        self.client_name = client_name
        self.num_classes = 62
        self.distribution = np.array(self.compute_distribution())
        self.distribution /= self.distribution.sum()
        self.entropy = self.compute_entropy()

    def __getitem__(self, index: int) -> Any:
        a = np.array(self.samples[index][0])
        a = np.reshape(a, (IMAGE_SIZE, IMAGE_SIZE, 1))
        b = np.array(self.samples[index][1])
        return self.transform(a), b

    def __len__(self) -> int:
        return len(self.samples)

    def compute_entropy(self): 
      """
      This method computes the dataset's entropy
      """
      holder = [-i*torch.log2(torch.tensor(i)) for i in self.distribution if i != 0]
      return torch.sum(torch.tensor(holder))
    
    def compute_distribution(self): 
      """
      This method comoputes the local distribution of the data
      """
      hold = [0] * self.num_classes
      for image, label in self.samples: 
              hold[label] += 1
      distribution = [i/len(self.samples) for i in hold]
      return distribution
