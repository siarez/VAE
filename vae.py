import torch
from torch.optim import Adam
from sklearn.datasets import fetch_mldata
from tqdm import tqdm
from logger import Logger

pz_size = 12
image_size = 28 * 28

x = torch.randn(image_size)
loss_func = torch.nn.MSELoss()


class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Input output Layer definitions
        self.input_layer = torch.nn.Linear(image_size, pz_size)
        self.activation_layer = torch.nn.Sigmoid()
        self.output_layer = torch.nn.Linear(int(pz_size / 2), image_size)

    def forward(self, x):
        mu_sigma = self.input_layer(x)
        mu, sigma = torch.chunk(mu_sigma, 2)

        # Calculating hidden representation `z`
        normal_sample = torch.autograd.Variable(
            torch.normal(torch.zeros(int(pz_size / 2)), torch.ones(int(pz_size / 2))),
            requires_grad=False)
        self.z = torch.mul(normal_sample, sigma) + mu

        # Calculating output
        return self.activation_layer(self.output_layer(self.z))


def _prepare_data():
    mnist = fetch_mldata('MNIST original', data_home='./')
    mnist_tensor = torch.ByteTensor(mnist.data).type(torch.FloatTensor)
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
    return loss_trace


logger = Logger("./logs")

vae = VAE()
optim = Adam(vae.parameters(), lr=0.001, weight_decay=0.001)  # lr: learning rate, weight decay: ?
mnist_tensor = _prepare_data()
input_var = torch.autograd.Variable(mnist_tensor)
trace = train(data_var=input_var)
# normal distribution definition
