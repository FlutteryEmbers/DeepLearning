import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from models.AE import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np

EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST= False
N_TEST_IMG = 5

train_data = torchvision.datasets.MNIST(
    root='./minist',
    train=True,
    transform=torchvision.transforms.ToTensor(),

    download=DOWNLOAD_MNIST,
)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

auto_encoder = AutoEncoder()

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())


for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, 28*28).to(auto_encoder.device)
        b_y = x.view(-1, 28*28).to(auto_encoder.device)

        encoded, decoded = auto_encoder(b_x)

        loss = auto_encoder.lose_func(decoded, b_y)

        auto_encoder.optimizer.zero_grad()
        loss.backward()
        auto_encoder.optimizer.step()

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())

            # plotting decoded image (second row)
            _, decoded_data = auto_encoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()
