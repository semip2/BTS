import numpy as np
import torch
import time
from torch import nn
from torch import optim
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision import datasets, transforms
from torch.nn.functional import softmax

# move the input and model to GPU for speed if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load in Alexnet model
torch.backends.cudnn.benchmark = True
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

# Set Criterion and Optimizer 
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.NLLLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Preprocess images: 224x224 RGB  
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

braille_train = datasets.ImageFolder(root='braille_train/', transform=preprocess)
braille_val = datasets.ImageFolder(root='braille_val/', transform=preprocess)
braille_test = datasets.ImageFolder(root='braille_test/', transform=preprocess)

batch_size = 32

train_loader = DataLoader(braille_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(braille_val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(braille_test, batch_size=batch_size, shuffle=True)

patience = 5
curr_count_to_patience = 0
global_min_loss = 999999999
epoch = 0
best_epoch = 0

save_path = "saves/"
save_name = "best_model.pth"

print("Training CNN...")
start = time.time()
while curr_count_to_patience < patience:
    # Train epoch
    running_loss_train = []
    for i, (X, y) in enumerate(train_loader):
        if i % 10 == 0:
            print(f"Training epoch {epoch}, batch {i}...\r", end="")
        # print(X)
        # print(y)
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = criterion(outputs, y)
        running_loss_train.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    loss_train = np.mean(running_loss_train)

    # Evaluate epoch on validation
    running_loss_valid = []
    with torch.no_grad():
        for (X, y) in val_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            running_loss_valid.append(criterion(output, y).item())
        
        loss_valid = np.mean(running_loss_valid)
        if loss_valid >= global_min_loss:
            curr_count_to_patience += 1
        else:
            global_min_loss = loss_valid
            curr_count_to_patience = 0
            # torch.save(model.state_dict(), f"{save_path}{save_name}")
            best_epoch = epoch
    print("------------------------------------------------")
    print(f"[INFO]: Epoch {epoch}")
    print(f"Training loss: {loss_train:0.3f}")
    print(f"Validation loss: {loss_valid:0.3f}")
    print("------------------------------------------------")
    epoch += 1

end = time.time()
print(f"Finished training after {epoch} epochs in {(end - start):0.2f} seconds")
print(f"Best epoch: {best_epoch}")

save_model = input(f"Save final model? (y/n): ")
if save_model == "y":
    torch.save(model.state_dict(), f"{save_path}final_model.pth")
    print(f"Best model saved to {save_path}{save_name}\n")
    print(f"Final model saved to {save_path}final_model.pth\n")

correct = 0
total = 0

print(f"Evaluating best model on test data...")
model.load_state_dict(torch.load(f"{save_path}{save_name}"))
with torch.no_grad():
    for (X, y) in test_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        _, predicted = torch.max(softmax(output.data, dim=1), 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(f'Accuracy of the best model on the test data: {100 * correct / total}%')
