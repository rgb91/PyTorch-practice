import torch
from torch import optim, nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from VAE.model import VariationalAutoEncoder

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200  # hidden dimension
Z_DIM = 20  # latent space dimension
NUM_EPOCHS = 5
BATCH_SIZE = 128
LEARN_RATE = 3e-4  # Karpathy Constant


def get_one_image_per_digit(dataset):
    idx = 0
    images = []
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break
    return images


def main():
    # Dataset
    dataset = datasets.MNIST(root="dataset/",
                             train=True,
                             transform=transforms.ToTensor(),  # div by 255
                             download=True)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = VariationalAutoEncoder(input_dim=INPUT_DIM, h_dim=H_DIM, z_dim=Z_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    # training
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))
        for i, (x, _) in loop:
            # forward pass
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
            x_reconstructed, mu, sigma = model(x)

            # loss calculation
            reconstruction_loss = loss_fn(x_reconstructed, x)
            # KL divergence is going to push toward standard Gaussian
            kl_divergence = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # back propagation
            loss = reconstruction_loss + kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    # inference
    num_of_examples = 3  # to generate per class
    images = get_one_image_per_digit(dataset=dataset)
    print(images[0].shape)
    encoded_images = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].to(DEVICE).view(1, 784))
            encoded_images.append((mu, sigma))

            for n in range(num_of_examples):
                epsilon = torch.rand_like(sigma)
                z = mu + sigma*epsilon
                out = model.decode(z)
                out = out.view(-1, 1, 28, 28)
                save_image(out, rf'./generated/gen_{d}_{n}.png')


if __name__ == '__main__':
    main()
