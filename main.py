import argparse
import os
import random
import scipy as sp
import pickle

import shutil
import csv
import ast

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample, q_sample, test_gaussian_properties
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset, get_diffusion_parameters, modified_beta_schedule


from torch.utils.data import Subset
np.random.seed(13)

"""
Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=2, help="Number of training epochs for the autoencoder (default: 200)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-encoder', type=int, default=64, help="Hidden dimension size for encoder layers (default: 64)")

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")

# Dimensionality of the latent space
parser.add_argument('--latent-dim', type=int, default=32, help="Dimensionality of the latent space in the autoencoder (default: 32)")

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

# Number of layers in the encoder network
parser.add_argument('--n-layers-encoder', type=int, default=2, help="Number of layers in the encoder network (default: 2)")

# Number of layers in the decoder network
parser.add_argument('--n-layers-decoder', type=int, default=3, help="Number of layers in the decoder network (default: 3)")

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=6, help="Number of training epochs for the denoising model (default: 100)")

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=500, help="Number of timesteps for the diffusion (default: 500)")

# Hidden dimension size for the denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512, help="Hidden dimension size for denoising model layers (default: 512)")

# Number of layers in the denoising model
parser.add_argument('--n-layers_denoise', type=int, default=3, help="Number of layers in the denoising model (default: 3)")

# Flag to toggle training of the autoencoder (VGAE)
parser.add_argument('--train-autoencoder', action='store_false', default=True, help="Flag to enable/disable autoencoder (VGAE) training (default: enabled)")

# Flag to toggle training of the diffusion-based denoising model
parser.add_argument('--train-denoiser', action='store_true', default=True, help="Flag to enable/disable denoiser training (default: enabled)")

# Dimensionality of conditioning vectors for conditional generation
parser.add_argument('--dim-condition', type=int, default=128, help="Dimensionality of conditioning vectors for conditional generation (default: 128)")

# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

##########
parser.add_argument('--train-gan', action='store_true', default=True,help="Flag to enable GAN-based adversarial training alongside autoencoder.")

parser.add_argument('--gan-lambda', type=float, default=0.1,help="Weight for the adversarial loss term (default: 0.1).")

parser.add_argument('--gan-n-critic', type=int, default=5,help="Number of critic/discriminator updates per generator update (WGAN) (default: 5).")

parser.add_argument('--lambda_gp', type=float, default=10.0, help='Gradient penalty coefficient')

parser.add_argument('--lr_disc', type=float, default=1e-4, help='Discriminator learning rate')
#########


args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)



# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

"""class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim=32):
        super(LatentDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)  # Critic score, not a probability
        )

    def forward(self, z):
        return self.model(z)


if args.train_gan:
    discriminator = LatentDiscriminator(latent_dim=args.latent_dim).to(device)
    print("ok")
    optimizer_disc = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.9)  # Commonly used betas for WGAN
    )"""

class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim=512):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.net(z)

def wgan_critic_loss(real_score, fake_score):
    """
    Computes the WGAN critic loss.
    """
    return fake_score.mean() - real_score.mean()

def wgan_gen_loss(fake_score):
    """
    Computes the WGAN generator loss.
    """
    return -fake_score.mean()

def gradient_penalty(discriminator, real_samples, fake_samples, device, gp_coef=10.0):
    """
    Computes the gradient penalty for WGAN-GP.
    """
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand_as(real_samples)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = gp_coef * ((gradient_norm - 1) ** 2).mean()
    return gp

# Initialize models
discriminator = Discriminator(latent_dim=args.latent_dim).to(device)
# initialize VGAE model
autoencoder = VariationalAutoEncoder(
    input_dim=args.spectral_emb_dim + 1,
    hidden_dim_enc=args.hidden_dim_encoder,
    latent_dim=args.latent_dim,
    stats_dim=args.n_condition,
    node_emb_dim=args.hidden_dim_encoder,  # Node embedding size in the decoder
    edge_hidden_dim=args.hidden_dim_decoder,  # Hidden dimension for edge MLP
    n_layers_enc=args.n_layers_encoder,
    n_layers_dec=args.n_layers_decoder,
    n_max_nodes=args.n_max_nodes
).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# Train VGAE model
if args.train_autoencoder:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder + 1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        train_loss_all_recon = 0
        train_loss_all_kld = 0
        train_loss_all_gan = 0
        cnt_train = 0

        if args.train_gan:
            discriminator.train()

        for data in train_loader:
            data = data.to(device)

            if args.train_gan:
                # Train Discriminator
                for _ in range(args.gan_n_critic):
                    optimizer_disc.zero_grad()

                    # Real latent samples from prior N(0, I)
                    real_z = torch.randn(data.num_graphs, args.latent_dim, device=device)

                    # Fake latent samples from encoder
                    x_g = autoencoder.encoder(data)
                    mu = autoencoder.fc_mu(x_g)
                    logvar = autoencoder.fc_logvar(x_g)
                    fake_z = autoencoder.reparameterize(mu, logvar)

                    # Compute critic scores
                    real_score = discriminator(real_z)
                    fake_score = discriminator(fake_z.detach())

                    # WGAN loss + gradient penalty
                    disc_loss = wgan_critic_loss(real_score, fake_score)
                    gp = gradient_penalty(discriminator, real_z.data, fake_z.data, device, args.lambda_gp)
                    disc_loss_total = disc_loss + gp

                    disc_loss_total.backward()
                    optimizer_disc.step()

            # Train Autoencoder
            optimizer.zero_grad()
            loss, recon, kld = autoencoder.loss_function(data, data.stats)

            if args.train_gan:
                # Get latent codes again for generator update
                x_g = autoencoder.encoder(data)
                mu = autoencoder.fc_mu(x_g)
                logvar = autoencoder.fc_logvar(x_g)
                fake_z = autoencoder.reparameterize(mu, logvar)

                # Get critic score for generator update
                fake_score_g = discriminator(fake_z)
                gen_loss = wgan_gen_loss(fake_score_g)

                # Combined loss
                total_loss = loss + args.gan_lambda * gen_loss
                train_loss_all_gan += gen_loss.item()
            else:
                total_loss = loss

            total_loss.backward()
            train_loss_all += loss.item()
            train_loss_all_recon += recon.item()
            train_loss_all_kld += kld.item()
            train_count += torch.max(data.batch) + 1
            cnt_train += 1
            optimizer.step()

        # Validation phase
        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_all_recon = 0
        val_loss_all_kld = 0

        for data in val_loader:
            data = data.to(device)
            loss, recon, kld = autoencoder.loss_function(data, data.stats)
            val_loss_all += loss.item()
            val_loss_all_recon += recon.item()
            val_loss_all_kld += kld.item()
            val_count += torch.max(data.batch) + 1
            cnt_val += 1

        # Print logging info
        dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log_str = '{} Epoch: {:03d} | Train Loss: {:.5f} (recon={:.2f}, kld={:.2f}'.format(
            dt_t, epoch,
            train_loss_all / cnt_train,
            train_loss_all_recon / cnt_train,
            train_loss_all_kld / cnt_train)

        if args.train_gan:
            log_str += f', gan={train_loss_all_gan / cnt_train:.2f}'

        log_str += f') | Val Loss: {val_loss_all / cnt_val:.5f} (recon={val_loss_all_recon / cnt_val:.2f}, kld={val_loss_all_kld / cnt_val:.2f})'
        print(log_str)

        scheduler.step()

        # Save best model
        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'discriminator': discriminator.state_dict() if args.train_gan else None,
                'optimizer_disc': optimizer_disc.state_dict() if args.train_gan else None,
            }, 'autoencoder.pth.tar')

else:
    checkpoint = torch.load('autoencoder.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])
    if args.train_gan and 'discriminator' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator'])

autoencoder.eval()

"""# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
"""
diff_params = get_diffusion_parameters(args.timesteps, beta_schedule="cosine")
betas = diff_params['betas']
alphas = diff_params['alphas']
alphas_cumprod = diff_params['alphas_cumprod']
alphas_cumprod_prev = diff_params['alphas_cumprod_prev']
sqrt_recip_alphas = diff_params['sqrt_recip_alphas']
sqrt_alphas_cumprod = diff_params['sqrt_alphas_cumprod']
sqrt_one_minus_alphas_cumprod = diff_params['sqrt_one_minus_alphas_cumprod']
posterior_variance = diff_params['posterior_variance']


# initialize denoising model
denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_condition, d_cond=args.dim_condition).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# Train denoising model
if args.train_denoiser:
    best_val_loss = np.inf
    current_timesteps = args.timesteps

    for epoch in range(1, args.epochs_denoise + 1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0

        # Recalculate diffusion parameters with current_timesteps
        diff_params = get_diffusion_parameters(current_timesteps, beta_schedule="cosine")
        betas = diff_params['betas']
        alphas = diff_params['alphas']
        alphas_cumprod = diff_params['alphas_cumprod']
        alphas_cumprod_prev = diff_params['alphas_cumprod_prev']
        sqrt_recip_alphas = diff_params['sqrt_recip_alphas']
        sqrt_alphas_cumprod = diff_params['sqrt_alphas_cumprod']
        sqrt_one_minus_alphas_cumprod = diff_params['sqrt_one_minus_alphas_cumprod']
        posterior_variance = diff_params['posterior_variance']

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)

            # Test Gaussian properties of fully noised latent
            t_final = torch.full((x_g.size(0),), current_timesteps - 1, device=device, dtype=torch.long)
            noise = torch.randn_like(x_g)
            z_T = q_sample(x_g, t_final, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)

            # Test if z_T is Gaussian
            test_results = test_gaussian_properties(z_T)

            # If not Gaussian enough and not at max timesteps, increase timesteps
            if (not test_results['is_gaussian'] or test_results['confidence_score'] < 0.2) and current_timesteps < 100000:
                current_timesteps = min(current_timesteps + 100, 100000)
                print(f"\nEpoch {epoch}: Increasing timesteps to {current_timesteps}")
                print(f"Gaussian test confidence: {test_results['confidence_score']:.4f}")
                # Recalculate diffusion parameters with new timesteps
                diff_params = get_diffusion_parameters(current_timesteps, beta_schedule="cosine")
                betas = diff_params['betas']
                alphas = diff_params['alphas']
                alphas_cumprod = diff_params['alphas_cumprod']
                alphas_cumprod_prev = diff_params['alphas_cumprod_prev']
                sqrt_recip_alphas = diff_params['sqrt_recip_alphas']
                sqrt_alphas_cumprod = diff_params['sqrt_alphas_cumprod']
                sqrt_one_minus_alphas_cumprod = diff_params['sqrt_one_minus_alphas_cumprod']
                posterior_variance = diff_params['posterior_variance']

            # Sample random timestep for training
            t = torch.randint(0, current_timesteps, (x_g.size(0),), device=device).long()

            # Calculate loss and update model
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod,
                            sqrt_one_minus_alphas_cumprod, loss_type="huber")
            loss.backward()
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()

        # Validation loop (using current_timesteps)
        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, current_timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod,
                            sqrt_one_minus_alphas_cumprod, loss_type="huber")
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}, Timesteps: {}'.format(
                dt_t, epoch, train_loss_all / train_count, val_loss_all / val_count, current_timesteps))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'current_timesteps': current_timesteps  # Save the final number of timesteps
            }, 'denoise_model.pth.tar')
else:
    checkpoint = torch.load('denoise_model.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])
    current_timesteps = checkpoint.get('current_timesteps', args.timesteps)  # Load saved timesteps

denoise_model.eval()

del train_loader, val_loader

# Save to a CSV file
with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
        data = data.to(device)
        
        stat = data.stats
        bs = stat.size(0)

        graph_ids = data.filename

        samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
        x_sample = samples[-1]
        adj = autoencoder.decode_mu(x_sample,stat)
        stat_d = torch.reshape(stat, (-1, args.n_condition))


        for i in range(stat.size(0)):
            stat_x = stat_d[i]

            Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
            stat_x = stat_x.detach().cpu().numpy()

            # Define a graph ID
            graph_id = graph_ids[i]

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])           
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text])