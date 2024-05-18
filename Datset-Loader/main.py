import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from datatset_loader import CatsAndDogsDataset

dataset = CatsAndDogsDataset(csv_file='data/train.csv',
     root_dir='data/',
     transform=transforms.toTensor())

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [20000,5000])

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")