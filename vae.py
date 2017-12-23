import torch
from torch.optim import Adam

pz_size = 30
image_size = 28 * 28

x = torch.randn(image_size)
loss_func = torch.nn.MSELoss()


class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Input output Layer definitions
        self.input_layer = torch.nn.Linear(image_size, pz_size)
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
        return self.output_layer(self.z)


# normal distribution definition
input_var = torch.autograd.Variable(x)
vae = VAE()
opim = Adam(vae.parameters(), lr=0.001, weight_decay=0.001)  # lr: learning rate, weight decay: ?
output = vae(input_var)  # forward pass
loss = loss_func(output, input_var)  # calculate loss
loss.backward()  # back propagate loss to calculate the deltas (the "gradient")
opim.step()  # use the learning rates and the gradient to update parameters

print(output)
