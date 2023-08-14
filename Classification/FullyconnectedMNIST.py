import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 50
batch_size = 100
learning_rate = 1e-3

#dataset
train_data = torchvision.datasets.MNIST(root = '/home/sasuke/Desktop/deep learning', train = True, download = True, transform = transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root = '/home/sasuke/Desktop/deep learning',transform = transforms.ToTensor())

#loader

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.l2(self.relu(self.l1(x)))
        return out
    

model = NN(input_size, hidden_size, num_classes).to(device)

cr = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_steps = len(train_loader)
for epochs in range(num_epochs):
    for _, (images, labels) in enumerate(train_loader):
        img = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        out = model(img)
        loss = cr(out, labels)        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            img = images.reshape(-1, 28 * 28).to(device)
            out = model(img)
            labels = labels.to(device)
            _, predictions = torch.max(out, dim = 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item() 
    print(f"epoch:{epochs + 1}, loss:{loss.item():.3f}, accuracy:{100 * n_correct/n_samples}")

        
