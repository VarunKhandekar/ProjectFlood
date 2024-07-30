import torch
import torch.nn as nn
import torch.optim as optim

def train_model(epochs):
    model.train()
    for epoch in range(epochs):  # number of epochs
        for images, labels in train_loader:
            # Move images and labels to the correct device and dtype
            images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')