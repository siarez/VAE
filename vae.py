import torch
from torch.optim import Adam
from sklearn.datasets import fetch_mldata
from tqdm import tqdm
from logger import Logger
import numpy as np

pz_size = 20
mid_size = 30
image_size = 28 * 28

x = torch.randn(image_size)
loss_func = torch.nn.MSELoss()


class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Input output Layer definitions
        self.input_layer = torch.nn.Linear(image_size, mid_size)
        self.middle_layer = torch.nn.Linear(mid_size, pz_size)
        self.activation_layer = torch.nn.Sigmoid()  # TODO: can this be re-used? looks like it
        self.middle_out_layer = torch.nn.Linear(int(pz_size / 2), mid_size)
        self.output_layer = torch.nn.Linear(mid_size, image_size)

    def forward(self, x):
        mu_sigma = self.activation_layer(self.middle_layer(self.input_layer(x)))
        mu, sigma = torch.chunk(mu_sigma, 2)

        # Calculating hidden representation `z`
        normal_sample = torch.autograd.Variable(
            torch.normal(torch.zeros(int(pz_size / 2)), torch.ones(int(pz_size / 2))),
            requires_grad=False)
        self.z = torch.mul(normal_sample, sigma) + mu

        # Calculating output
        return self.activation_layer(self.output_layer(self.activation_layer(self.middle_out_layer(self.z))))


def _prepare_data():
    mnist = fetch_mldata('MNIST original', data_home='./')
    mnist_tensor = torch.ByteTensor(mnist.data).type(torch.FloatTensor)
    rnd_idx = np.random.choice(len(mnist.data), len(mnist.data), replace=False)  # pick index values randomly
    mnist_tensor = mnist_tensor[rnd_idx, :]  # randomize the data set
    mnist_tensor = mnist_tensor / 255  # normalize
    print(mnist_tensor.shape)
    return mnist_tensor


def train(data_var, epochs=5):
    loss_trace = torch.zeros(epochs, data_var.shape[0])

    for i in tqdm(range(epochs)):
        for j in tqdm(range(data_var.shape[0])):
            output = vae(data_var[j][:])  # forward pass
            loss = loss_func(output, data_var[j][:])  # calculate loss
            loss_trace[i][j] = loss.data[0]
            logger.scalar_summary("loss", loss.data[0], i + 1)
            loss.backward()  # back propagate loss to calculate the deltas (the "gradient")
            optim.step()  # use the learning rates and the gradient to update parameters
            if j % 5000 == 0:
                log_images(data_var[j][:], output, i + 1)

    return loss_trace


def log_images(input, output, epoch):
    reshaped_in = input.view(28, 28).data.numpy()
    reshaped_out = output.view(28, 28).data.numpy()
    logger.image_summary('prediction', [reshaped_in, reshaped_out], epoch)


logger = Logger("./logs")

vae = VAE()
optim = Adam(vae.parameters(), lr=0.00001, weight_decay=0.001)  # lr: learning rate, weight decay: ?
mnist_tensor = _prepare_data()
input_var = torch.autograd.Variable(mnist_tensor)
trace = train(data_var=input_var)
# normal distribution definition
