from model import UNet, criterion, optimizer
from data import train_generator, test_generator, save_result, train_generator_2, evaluateRes, test_generator2
import torch
from data_assign import data_assign
# Testing

train_path = 'dataset/train'
test_path = 'dataset/test'
image_path = "dataset/CXR_png"
mask_path = "dataset/mask"
model_save_path = 'unet_membrane.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet()
model.load_state_dict(torch.load(model_save_path))
model.eval()
test_loader = test_generator2(test_path, num_image=114)
results = []

with torch.no_grad():

    images_list = []
    while True:
        try:
            images = next(test_loader)
            images_list.append(images)
        except StopIteration:
            break
    print("images_list", len(images_list))

    for images in images_list:
        images = images.to(device)
        outputs = model(images)
        results.append(outputs.cpu().numpy())

# print("results", results)

# Save results
# save_result(test_path, results)
rate_correct = evaluateRes(test_path, results)
print("rate_correct", rate_correct)
