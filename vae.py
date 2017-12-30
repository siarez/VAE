import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, sampler
from sklearn.datasets import fetch_mldata
from tqdm import tqdm
from logger import Logger
import numpy as np
import json

EPOCHS = 6
BATCH_SIZE = 35
pz_size = 20
mid_size = 100
image_size = 28 * 28
# Check for CUDA
cuda = torch.cuda.is_available()


class VAE(torch.nn.Module):
    """VAE model"""
    def __init__(self):
        super(VAE, self).__init__()
        # Input output Layer definitions
        self.encoder_input = torch.nn.Linear(image_size, mid_size)
        self.encoder_hidden = torch.nn.Linear(mid_size, pz_size)
        self.sigmoid = torch.nn.Sigmoid()  # TODO: can this be re-used? looks like it
        self.relu = torch.nn.ReLU()
        self.decoder_hidden = torch.nn.Linear(int(pz_size / 2), mid_size)
        self.decoder_output = torch.nn.Linear(mid_size, image_size)
        self.normal_tensor = torch.zeros(BATCH_SIZE, int(pz_size / 2))
        if cuda:
            self.normal_tensor.cuda()
        self.z = None

    def forward(self, x):
        return self.forward_variational(x)
        # return self.forward_vanilla(x)

    def forward_variational(self, x):
        mu_logvar = self.encoder_hidden(self.relu(self.encoder_input(x)))
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
        std = logvar.mul(0.5).exp()

        # Calculating hidden representation `z`
        self.normal_tensor.normal_()
        self.z = torch.mul(torch.autograd.Variable(self.normal_tensor, requires_grad=False), std) + mu
        # Calculating output
        return self.sigmoid(self.decoder_output(self.relu(self.decoder_hidden(self.z)))), mu, logvar

    def forward_vanilla(self, x):
        """Forward function for a regular vanilla autoencoder"""
        pz = self.relu(self.encoder_hidden(self.relu(self.encoder_input(x))))
        # We cut pz in half
        self.z, _ = torch.chunk(pz, 2, dim=1)
        # Calculating output
        return self.sigmoid(self.decoder_output(self.relu(self.decoder_hidden(self.z)))), self.z, None

    def save_json(self):
        """
        Saves model parameters in JSON for use in deeplearning.js
        :return: none
        """
        model_params = {
            "enc_in_weight": {"param": self.encoder_input.weight.data.numpy().tolist(), "dim": self.encoder_input.weight.data.numpy().shape},
            "enc_in_bias": {"param": self.encoder_input.bias.data.numpy().tolist(), "dim": self.encoder_input.bias.data.numpy().shape},
            "enc_hid_weight": {"param": self.encoder_hidden.weight.data.numpy().tolist(), "dim": self.encoder_hidden.weight.data.numpy().shape},
            "enc_hid_bias": {"param": self.encoder_hidden.bias.data.numpy().tolist(), "dim": self.encoder_hidden.bias.data.numpy().shape},
            "dec_hid_weight": {"param": self.decoder_hidden.weight.data.numpy().tolist(), "dim": self.decoder_hidden.weight.data.numpy().shape},
            "dec_hid_bias": {"param": self.decoder_hidden.bias.data.numpy().tolist(), "dim": self.decoder_hidden.bias.data.numpy().shape},
            "dec_out_weight": {"param": self.decoder_output.weight.data.numpy().tolist(), "dim": self.decoder_output.weight.data.numpy().shape},
            "dec_out_bias": {"param": self.decoder_output.bias.data.numpy().tolist(), "dim": self.decoder_output.bias.data.numpy().shape}
        }
        json_params = json.dumps(model_params)
        text_file = open("model_params.json", "w")
        text_file.write(json_params)
        text_file.close()


class MyData(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, item):
        return self.input[item]

    def get_sample_digit(self, digit):
        """A very simple function to retrieve a random mnist samples of the desired digit."""
        sample_size = 100  # 100 should be big enough of a sample size to contain at least one of every digit.
        random_offset = np.random.randint(0, len(self) - sample_size)
        for idx in range(random_offset, random_offset + sample_size):
            if self.target[idx] == digit:
                return self.input[idx]
        raise ValueError


def kl_loss(mu, logvar):
    """
    Calculates the KL divergence component of the loss function
    :param mu: the mean/s of the generated distribution by the encoder
    :param logvar: the log(variance)/s of the generated distribution by encoder
    :return: The sum of KL divergences between the distributions generated by the encoder and
    the “standard normal distribution” i.e. N(0, 1)
    """
    # Todo: this scaling factor was used by the implementation in pytorch repo's VAE example.
    # Todo: Not sure of thr reasoning behind it.
    scaling_factor = (logvar.shape[0] * image_size)
    KL_dive = -0.5*torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    return KL_dive / scaling_factor


def _prepare_data():
    """Downloads (if necessary), randomizes and prepares the dataset."""
    mnist = fetch_mldata('MNIST original', data_home='./')
    mnist_tensor = torch.ByteTensor(mnist.data).type(torch.FloatTensor)
    mnist_target = torch.ByteTensor(mnist.target)
    if cuda:
        mnist_tensor.cuda()
    rnd_idx = np.random.choice(len(mnist.data), len(mnist.data), replace=False)  # pick index values randomly
    mnist_tensor = mnist_tensor[rnd_idx, :]  # randomize the data set
    mnist_target = mnist_target[torch.LongTensor(rnd_idx)]  # randomize the data set
    mnist_tensor /= 255  # normalize
    dataset = MyData(mnist_tensor, mnist_target)
    print(mnist_tensor.shape)
    return dataset


def log_images(tag, img, reconstruction, epoch):
    reshaped_in = img.view(28, 28).data.cpu().numpy()
    reshaped_out = reconstruction.view(28, 28).data.cpu().numpy()
    logger.image_summary(tag, [reshaped_in, reshaped_out], epoch)


def train(data_set, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Trains the model with the `data_set`"""
    training_size = 60000  # Whatever the train size, the remainder (70000 - training_size) will be used for validation
    # Create random indices
    rnd_idx = np.random.choice(len(data_set), len(data_set), replace=False)  # pick index values randomly

    data_loader = DataLoader(data_set, batch_size=batch_size,
                             sampler=sampler.SubsetRandomSampler(rnd_idx[:training_size]), drop_last=True)
    val_data_loader = DataLoader(data_set, batch_size=batch_size,
                                 sampler=sampler.SubsetRandomSampler(rnd_idx[training_size:]), drop_last=True)
    loss_func = torch.nn.MSELoss()
    for epoch in tqdm(range(epochs)):
        for ii, batch_raw in enumerate(tqdm(data_loader)):
            optim.zero_grad()
            batch = torch.autograd.Variable(batch_raw)
            output, mu, logvar = vae(batch)  # forward pass
            # calculate loss. We don't calculate KL_loss for vanilla autoencoder (i.e. logvar is None)
            loss = loss_func(output, batch) + (kl_loss(mu, logvar) if logvar is not None else 0)
            logger.scalar_summary("loss", loss.data[0], epoch * len(data_loader) + ii)
            loss.backward()  # back propagate loss to calculate the deltas (the "gradient")
            optim.step()  # use the learning rates and the gradient to update parameters
            if ii % 100 == 0 and False:
                # logging images for viewing Tensorboard
                log_images("Training", batch[0], output[0], ii)
                '''
                # running the validation dataset
                val_loss = 0
                num_of_batches = 0
                for val_image in tqdm(val_data_loader):
                    val_image_var = torch.autograd.Variable(val_image)
                    output, mu, logvar = vae(val_image_var)  # forward pass
                    val_loss += loss_func(output, val_image_var) + kl_loss(mu, logvar)  # calculate loss TODO: Should we add the regularizer term for validation?
                    num_of_batches += 1
                log_images("Validation", val_image_var[0], output[0], ii)
                logger.scalar_summary("val_loss", val_loss.data[0]/num_of_batches, epoch * len(data_loader) + ii)
                '''



logger = Logger("./logs")
vae = VAE()

if cuda:
    vae.cuda()
optim = Adam(vae.parameters(), lr=0.001)  # lr: learning rate, weight decay: ?
mnist_dataset = _prepare_data()
train(data_set=mnist_dataset)
vae.save_json()
# Get some sample images and their z representation for the blog
z_representations = []
for digit in range(0, 10):
    image = mnist_dataset.get_sample_digit(digit)
    image.unsqueeze_(0)
    output, mu, logvar = vae(torch.autograd.Variable(image))
    z_representations.append(mu.data.numpy().tolist()[0])
print(json.dumps(z_representations))

