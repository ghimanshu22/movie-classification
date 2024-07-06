import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertTokenizer

class MovieClassificationModel(nn.Module):
    def __init__(self):
        super(MovieClassificationModel, self).__init__()
        
        # Text branch (BERT)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(768, 256)
        
        # Image branch (ResNet)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)
        
        # Combined branch
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 10)  # Adjust the number of classes as needed
        
    def forward(self, text, images):
        # Text processing
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        text_features = self.bert(**tokens).last_hidden_state[:, 0, :]
        text_features = self.text_fc(text_features)
        
        # Image processing
        image_features = self.resnet(images)
        
        # Combine features
        combined_features = torch.cat((text_features, image_features), dim=1)
        x = torch.relu(self.fc1(combined_features))
        x = self.fc2(x)
        
        return x
