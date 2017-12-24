import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_mldata
from tqdm import tqdm
from logger import Logger
import numpy as np

BATCH_SIZE = 3
pz_size = 40
mid_size = 200
image_size = 28 * 28

x = torch.randn(image_size)
loss_func = torch.nn.MSELoss()
#TODO: Check for CUDA
dtype_var = torch.cuda.FloatTensor

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Input output Layer definitions
        self.input_layer = torch.nn.Linear(image_size, mid_size)
        self.middle_layer = torch.nn.Linear(mid_size, pz_size)
        self.activation_layer = torch.nn.Sigmoid()  # TODO: can this be re-used? looks like it
        self.relu_activate = torch.nn.ReLU()
        self.middle_out_layer = torch.nn.Linear(int(pz_size / 2), mid_size)
        self.output_layer = torch.nn.Linear(mid_size, image_size)
        self.normal_tensor = torch.zeros(int(pz_size/2)).cuda()

    def forward(self, x):
        mu_sigma = self.relu_activate(self.middle_layer(self.relu_activate(self.input_layer(x))))
        mu, logvar = torch.chunk(mu_sigma, 2, dim=1)
        std = logvar.mul(0.5).exp()

        # Calculating hidden representation `z`
        # normal_sample = torch.autograd.Variable(
        #     torch.normal(torch.zeros(int(pz_size / 2)), torch.ones(int(pz_size / 2))),
        #     requires_grad=False).cuda()
        self.normal_tensor.normal_()
        # self.z = torch.mul(normal_sample, std) + mu
        self.z = torch.mul(torch.autograd.Variable(self.normal_tensor, requires_grad=False), std) + mu
        # Calculating output
        return self.activation_layer(self.output_layer(self.relu_activate(self.middle_out_layer(self.z)))), mu, logvar

    def forward_classical(self, x):
        mu_sigma = self.relu_activate(self.middle_layer(self.input_layer(x)))
        mu, sigma = torch.chunk(mu_sigma, 2)
        self.z = mu
        # Calculating output
        return self.activation_layer(self.output_layer(self.relu_activate(self.middle_out_layer(self.z))))

class MyData(Dataset):
    def __init__(self, input):
        self.input = input

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, item):
        return self.input[item]


def kl_loss(mu, logvar):
    return -0.5*torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))

def _prepare_data():
    mnist = fetch_mldata('MNIST original', data_home='./')
    mnist_tensor = torch.ByteTensor(mnist.data).type(dtype_var)
    rnd_idx = np.random.choice(len(mnist.data), len(mnist.data), replace=False)  # pick index values randomly
    mnist_tensor = mnist_tensor[rnd_idx, :]  # randomize the data set
    mnist_tensor = mnist_tensor / 255  # normalize

    dataset = MyData(mnist_tensor)
    print(mnist_tensor.shape)
    return dataset


def train(data_set, epochs=5):
    # loss_trace = torch.zeros(epochs, len(data_set))
    loss_trace = []
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE)
    for i in tqdm(range(epochs)):
        for batch_raw in tqdm(data_loader):
            batch = torch.autograd.Variable(batch_raw)
            output, mu, logvar = vae(batch)  # forward pass
            loss = loss_func(output, batch) + kl_loss(mu, logvar) # calculate loss
            loss_trace.append(torch.sum(loss.data))
            logger.scalar_summary("loss", loss.data[0], i + 1)
            loss.backward()  # back propagate loss to calculate the deltas (the "gradient")
            optim.step()  # use the learning rates and the gradient to update parameters
            # if j % 5000 == 0:
            #     log_images(batch.data[], output, (j+1) + data_var.shape[0]*(i + 1))

    return loss_trace


def log_images(input, output, epoch):
    reshaped_in = input.view(28, 28).data.cpu().numpy()
    reshaped_out = output.view(28, 28).data.cpu().numpy()
    logger.image_summary('prediction', [reshaped_in, reshaped_out], epoch)


logger = Logger("./logs")

vae = VAE().cuda()
optim = Adam(vae.parameters(), lr=0.00001)  # lr: learning rate, weight decay: ?
mnist_dataset = _prepare_data()
trace = train(data_set=mnist_dataset)
# normal distribution definition
