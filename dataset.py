import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MovieDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['description']
        image_path = self.data.iloc[idx]['poster_path']
        label = self.data.iloc[idx]['genre']  # Map genres to numeric labels
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return text, image, label
