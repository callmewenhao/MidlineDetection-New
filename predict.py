from PIL import Image
import torch
import torchvision.transforms as transforms
from model import ResNet18
from utils import show_landmarks

# path etc.
img_path = "data/test_images/1.png"
weight_save_path = "midline.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# model
model = ResNet18(1, 10).to(device)
model.load_state_dict(torch.load(weight_save_path))
model.eval()

# image
image = Image.open(img_path)
image_data = transform(image).to(device).unsqueeze(0)
# print(image.shape)
with torch.no_grad():
    pred = model(image_data)
    # print(pred.shape)

# plot predictions
landmarks = pred[0].cpu().numpy()
show_landmarks(image, landmarks[:5]*94, landmarks[5:]*60)





