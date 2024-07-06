import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.text_image_model import MovieClassificationModel
from dataset import MovieDataset

# Initialize dataset, dataloader, model, loss function, and optimizer
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = MovieDataset('data/movie_data_with_posters.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MovieClassificationModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for texts, images, labels in dataloader:
        outputs = model(texts, images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), 'models/movie_classification_model.pth')
