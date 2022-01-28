from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import MidlineDataset
from model import ResNet18


# Hyper Parameters etc.
batch_size = 16
learning_rate = 1e-4
epochs = 150

img_dir = "data/train_images"
txt_file = "data/train.txt"
weight_save_path = "midline.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
dataset = MidlineDataset(img_dir, txt_file)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# model
model = ResNet18(1, 10).to(device)

# optim & loss_fn
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 每5个epoch检查一次loss, 如果loss不下降，就减小lr到0.5*lr
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

loss_fn = nn.MSELoss()

for epoch in range(epochs):
    losses = []
    loop = tqdm(dataloader)
    for idx, (image, label) in enumerate(loop):
        image = image.to(device)
        label = label.to(device)
        out = model(image)
        loss = loss_fn(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # progress bar
        loop.set_description(f"Epoch:{epoch}/{epochs}")
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    lr_scheduler.step(mean_loss)

# save model weights
torch.save(model.state_dict(), weight_save_path)



