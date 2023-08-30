import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from model import UNet, criterion, optimizer
from data import train_generator, test_generator, save_result
import torch
from data_assign import data_assign

# data set: https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities

train_path = 'dataset/train'
test_path = 'dataset/test'
image_path = "dataset/CXR_png"
mask_path = "dataset/mask"

train_mask_list, test_mask_list = data_assign(train_path, test_path, image_path, mask_path, rate=0.8)

print("number of training data", len(train_mask_list))
print("number of testing data", len(test_mask_list))

# Parameters
data_gen_args = dict(horizontal_flip=0.5)  # Only horizontal_flip was implemented in train_generator
model_save_path = 'unet_membrane.pth'
batch_size = 2
epochs = 1
steps_per_epoch = 300

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data generator
train_loader = train_generator(batch_size, train_path, 'image', 'label', data_gen_args)
# Initialize U-Net model
model = UNet().to(device)
# Define checkpoint logic
best_loss = float('inf')


def save_checkpoint():
    torch.save(model.state_dict(), model_save_path)
    print("Checkpoint saved!")

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    print("Epoch {}/{}".format(epoch + 1, epochs))

    for i, (images, masks) in enumerate(train_loader):
        print("start", i, images.shape, masks.shape)
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        # loss = nn.BCELoss()(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print("the average loss of", i, ":", total_loss/(i+1))

        if (i + 1) % steps_per_epoch == 0:
            average_loss = total_loss / steps_per_epoch
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{steps_per_epoch}], Loss: {average_loss:.4f}")

            # Save model if loss improves
            if average_loss < best_loss:
                best_loss = average_loss
                save_checkpoint()

            total_loss = 0.0

# Save model
torch.save(model.state_dict(), model_save_path)
model.load_state_dict(torch.load(model_save_path))

# Testing
model.eval()
test_loader = test_generator(test_path, num_image=57)
results = []
with torch.no_grad():
    for i in range(len(test_mask_list)):
        images = next(test_loader).to(device)
        outputs = model(images)
        results.append(outputs.cpu().numpy())
        print("test", i, outputs.shape)


# Save results
save_result(test_path, results)
