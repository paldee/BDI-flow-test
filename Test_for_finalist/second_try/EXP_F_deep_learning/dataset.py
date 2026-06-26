import torch
from torch.utils.data import Dataset
import numpy as np

class NMRDataset(Dataset):
    def __init__(self, spectra_matrix, labels_matrix):
        """
        spectra_matrix: (N, 10001) np.ndarray, float32
        labels_matrix: (N, 1328) np.ndarray, float32 (or int)
        """
        self.spectra = torch.from_numpy(spectra_matrix).float()
        self.labels = torch.from_numpy(labels_matrix).float()
        
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        # Return (1, 10001) for 1D-CNN input and (1328) for labels
        spec = self.spectra[idx].unsqueeze(0)
        label = self.labels[idx]
        return spec, label
