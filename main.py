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

from autoencoder import *
from denoise_model import DenoiseNN, p_losses, sample
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset
from torch_geometric.utils import to_dense_batch, to_dense_adj

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
parser.add_argument('--epochs-autoencoder', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")

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
parser.add_argument('--epochs-denoise', type=int, default=100, help="Number of training epochs for the denoising model (default: 100)")

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




"""
# initialize VGAE model
autoencoder = VariationalAutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)


# Train VGAE model
if args.train_autoencoder:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder+1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        train_loss_all_recon = 0
        train_loss_all_kld = 0
        cnt_train=0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, recon, kld  = autoencoder.loss_function(data)
            train_loss_all_recon += recon.item()
            train_loss_all_kld += kld.item()
            cnt_train+=1
            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch)+1
            optimizer.step()

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_all_recon = 0
        val_loss_all_kld = 0

        for data in val_loader:
            data = data.to(device)
            loss, recon, kld  = autoencoder.loss_function(data)
            val_loss_all_recon += recon.item()
            val_loss_all_kld += kld.item()
            val_loss_all += loss.item()
            cnt_val+=1
            val_count += torch.max(data.batch)+1

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(dt_t,epoch, train_loss_all/cnt_train, train_loss_all_recon/cnt_train, train_loss_all_kld/cnt_train, val_loss_all/cnt_val, val_loss_all_recon/cnt_val, val_loss_all_kld/cnt_val))
            
        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'autoencoder.pth.tar')
else:
    checkpoint = torch.load('autoencoder.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()"""

#########################################################################

class MultiScaleVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim_global,
        hidden_dim_local,
        latent_dim_global,
        latent_dim_local,
        num_nodes,
        hidden_dim_dec
    ):
        super(MultiScaleVAE, self).__init__()
        # -----------------------------
        # 1) Global & local encoders
        #    (e.g., DiffPool-based & GAT-based)
        # -----------------------------
        self.global_encoder = GlobalEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim_global,
            num_nodes=num_nodes,
            assign_ratio=0.25
        )
        self.local_encoder = LocalEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim_local
        )

        # -----------------------------
        # 2) Latent space projections
        # -----------------------------
        self.fc_mu_global = nn.Linear(hidden_dim_global, latent_dim_global)
        self.fc_logvar_global = nn.Linear(hidden_dim_global, latent_dim_global)
        self.fc_mu_local = nn.Linear(hidden_dim_local, latent_dim_local)
        self.fc_logvar_local = nn.Linear(hidden_dim_local, latent_dim_local)

        # -----------------------------
        # 3) Decoder (example)
        # -----------------------------
        self.decoder = Decoder(latent_dim_global + latent_dim_local,
                               hidden_dim_dec,
                               args.n_layers_decoder,
                               args.n_max_nodes)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, data):
        # For the global encoder, we assume data.x_dense & data.adj_dense are prepared
        x_global = self.global_encoder(data.x_dense, data.adj_dense)
        # For the local encoder, we assume data.x, data.edge_index, data.batch
        x_local = self.local_encoder(data.x, data.edge_index, data.batch)

        mu_g = self.fc_mu_global(x_global)
        logvar_g = self.fc_logvar_global(x_global)
        z_global = self.reparameterize(mu_g, logvar_g)

        mu_l = self.fc_mu_local(x_local)
        logvar_l = self.fc_logvar_local(x_local)
        z_local = self.reparameterize(mu_l, logvar_l)

        return z_global, mu_g, logvar_g, z_local, mu_l, logvar_l

    def decode(self, z_global, z_local):
        z = torch.cat([z_global, z_local], dim=-1)
        adj_hat = self.decoder(z)
        return adj_hat

    def forward(self, data):
        z_global, mu_g, logvar_g, z_local, mu_l, logvar_l = self.encode(data)
        adj_hat = self.decode(z_global, z_local)
        return adj_hat, (mu_g, logvar_g, mu_l, logvar_l)

    def loss_function(self, data, beta=0.01):
        z_global, mu_g, logvar_g, z_local, mu_l, logvar_l = self.encode(data)
        adj_hat = self.decode(z_global, z_local)

        # Example reconstruction loss
        recon_loss = F.l1_loss(adj_hat, data.A, reduction='mean')

        # KLD terms
        kld_global = -0.5 * torch.mean(1 + logvar_g - mu_g.pow(2) - logvar_g.exp())
        kld_local = -0.5 * torch.mean(1 + logvar_l - mu_l.pow(2) - logvar_l.exp())
        kld = kld_global + kld_local

        loss = recon_loss + beta * kld
        return loss, recon_loss, kld

in_channels = trainset[0].x.size(1)  # if x is [N, feats]

model = MultiScaleVAE(
    in_channels       = in_channels,
    hidden_dim_global = args.hidden_dim_encoder,  # or separate if you want
    hidden_dim_local  = args.hidden_dim_encoder,  # re-use the same dimension
    latent_dim_global = args.latent_dim,          # e.g. 32
    latent_dim_local  = args.latent_dim,          # e.g. also 32
    num_nodes         = args.n_max_nodes,         # e.g. 50
    hidden_dim_dec    = args.hidden_dim_decoder   # e.g. 256
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

if args.train_autoencoder:
    best_val_loss = np.inf

    for epoch in range(1, args.epochs_autoencoder + 1):
        model.train()
        train_loss_all = 0.0
        cnt_train = 0

        for data in train_loader:
            data = data.to(device)
            # Convert to dense for global encoder
            data.x_dense, _ = to_dense_batch(data.x, data.batch)
            data.adj_dense = to_dense_adj(data.edge_index, data.batch)

            optimizer.zero_grad()
            loss, recon, kld = model.loss_function(data, beta=0.01)
            loss.backward()
            optimizer.step()

            train_loss_all += loss.item()
            cnt_train += 1

        train_loss_epoch = train_loss_all / cnt_train

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss_all = 0.0
        cnt_val = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x_dense, _ = to_dense_batch(data.x, data.batch)
                data.adj_dense = to_dense_adj(data.edge_index, data.batch)

                loss, recon, kld = model.loss_function(data, beta=0.01)
                val_loss_all += loss.item()
                cnt_val += 1

        val_loss_epoch = val_loss_all / cnt_val

        scheduler.step()

        dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"{dt_t} [Epoch {epoch:03d}] "
              f"Train Loss: {train_loss_epoch:.4f} | "
              f"Val Loss: {val_loss_epoch:.4f}")

        # Save checkpoint if this is best so far
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'multi_scale_vae_best.pth.tar')
else:
    # Load the previously saved checkpoint if not training
    if os.path.isfile('multi_scale_vae_best.pth.tar'):
        checkpoint = torch.load('multi_scale_vae_best.pth.tar', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded MultiScaleVAE from checkpoint.")
    else:
        print("No checkpoint found. Model is uninitialized.")


# Example beta schedule
def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


# Simple denoising model,
# now input_dim = (latent_dim_global + latent_dim_local)
class DenoiseNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=2, n_cond=0, d_cond=0):
        super(DenoiseNN, self).__init__()
        # Example: MLP stack
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, input_dim))  # predict noise
        self.model = nn.Sequential(*layers)

        # If you have condition embeddings, you can incorporate them as well:
        # e.g. a separate MLP to embed condition -> concat -> pass through network
        # For simplicity, ignoring that here.

    def forward(self, x):
        return self.model(x)


# p_losses function (example placeholder)
def p_losses(denoise_model, x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber"):
    """
    x_0: [B, D_latent], the original (clean) latent
    t:   [B], a random integer time step
    ...
    returns a scalar loss
    """
    # get noisy x_t
    #   alpha_t = sqrt_alphas_cumprod[t], etc.
    # compute model's predicted noise
    # measure difference.
    # This is the part you already have in your prior code. Just ensure dimensional consistency.
    # For brevity, let's do a fake placeholder:

    B = x_0.size(0)
    device = x_0.device
    # gather sqrt_alphas_cumprod[t] for each in batch
    alpha_t = sqrt_alphas_cumprod[t].unsqueeze(-1)  # [B,1]
    alpha_t = alpha_t.to(device)

    noise = torch.randn_like(x_0)
    x_t = alpha_t * x_0 + (1. - alpha_t ** 2).sqrt() * noise

    # predict noise with denoise_model
    noise_pred = denoise_model(x_t)

    if loss_type == "huber":
        loss = F.smooth_l1_loss(noise_pred, noise)
    else:
        loss = F.mse_loss(noise_pred, noise)
    return loss


# sample function
@torch.no_grad()
def sample(denoise_model, batch_size, D_latent, timesteps, betas, device):
    """
    Return the entire trajectory [x_t for t in range(timesteps)],
    so you can pick x_0 at the end or any intermediate state.
    """
    x_t = torch.randn(batch_size, D_latent, device=device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - torch.cumprod(1. - betas, dim=0))
    sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(1. - betas, dim=0))

    for i in reversed(range(timesteps)):
        # i = t
        beta_t = betas[i]
        alpha_t = 1. - beta_t
        alpha_cumprod_t = torch.cumprod(1. - betas[:i + 1], dim=0)[-1]
        alpha_cumprod_t_prev = torch.cumprod(1. - betas[:i], dim=0)[-1] if i > 0 else torch.tensor(1., device=device)
        sqrt_one_minus_alpha_cumprod_t = (1 - alpha_cumprod_t).sqrt()

        # predict noise
        noise_pred = denoise_model(x_t)

        # eq (11) in DDPM paper, with variance
        if i > 0:
            posterior_variance_t = beta_t * (1. - alpha_cumprod_t_prev) / (1. - alpha_cumprod_t)
            # sample from x_{t-1}
            x_t = (1. / (alpha_t.sqrt())) * (x_t - (1. - alpha_t) * noise_pred / sqrt_one_minus_alpha_cumprod_t) \
                  + (posterior_variance_t.sqrt()) * torch.randn_like(x_t)
        else:
            # final step: directly predict x_0
            x_t = (1. / (alpha_t.sqrt())) * (x_t - (1. - alpha_t) * noise_pred / sqrt_one_minus_alpha_cumprod_t)
    return x_t

D_latent = args.latent_dim * 2  # since we have global+local
denoise_model = DenoiseNN(
    input_dim=D_latent,
    hidden_dim=args.hidden_dim_denoise,
    n_layers=args.n_layers_denoise
).to(device)

optimizer_denoise = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler_denoise = torch.optim.lr_scheduler.StepLR(optimizer_denoise, step_size=500, gamma=0.1)

timesteps = args.timesteps
betas = linear_beta_schedule(timesteps=timesteps).to(device)  # or keep on CPU, your choice

best_val_loss = np.inf

if args.train_denoiser:
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0.0
        train_count = 0

        for data in train_loader:
            data = data.to(device)
            # 1) encode with MultiScaleVAE -> get (z_global, z_local)
            data.x_dense, _ = to_dense_batch(data.x, data.batch)
            data.adj_dense = to_dense_adj(data.edge_index, data.batch)
            with torch.no_grad():
                z_g, _, _, z_l, _, _ = model.encode(data)
            x_0 = torch.cat([z_g, z_l], dim=-1)  # shape [B, D_latent]

            # 2) pick a random time t
            t = torch.randint(0, timesteps, (x_0.size(0),), device=device).long()

            # 3) compute p_losses
            loss = p_losses(
                denoise_model,
                x_0,
                t,
                sqrt_alphas_cumprod=torch.sqrt(torch.cumprod(1. - betas, axis=0)),
                sqrt_one_minus_alphas_cumprod=torch.sqrt(1. - torch.cumprod(1. - betas, axis=0)),
                loss_type="huber"
            )

            optimizer_denoise.zero_grad()
            loss.backward()
            optimizer_denoise.step()

            train_loss_all += loss.item() * x_0.size(0)
            train_count += x_0.size(0)

        denoise_model.eval()
        val_loss_all = 0.0
        val_count = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x_dense, _ = to_dense_batch(data.x, data.batch)
                data.adj_dense = to_dense_adj(data.edge_index, data.batch)
                with torch.no_grad():
                    z_g, _, _, z_l, _, _ = model.encode(data)
                x_0 = torch.cat([z_g, z_l], dim=-1)

                t = torch.randint(0, timesteps, (x_0.size(0),), device=device).long()
                val_loss = p_losses(
                    denoise_model,
                    x_0,
                    t,
                    sqrt_alphas_cumprod=torch.sqrt(torch.cumprod(1. - betas, axis=0)),
                    sqrt_one_minus_alphas_cumprod=torch.sqrt(1. - torch.cumprod(1. - betas, axis=0)),
                    loss_type="huber"
                )
                val_loss_all += val_loss.item() * x_0.size(0)
                val_count += x_0.size(0)

        train_loss_epoch = train_loss_all / train_count
        val_loss_epoch = val_loss_all / val_count

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"{dt_t} Epoch: {epoch:03d}, Train Loss: {train_loss_epoch:.5f}, Val Loss: {val_loss_epoch:.5f}")

        scheduler_denoise.step()

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer': optimizer_denoise.state_dict(),
            }, 'denoise_model.pth.tar')
else:
    if os.path.isfile('denoise_model.pth.tar'):
        checkpoint = torch.load('denoise_model.pth.tar')
        denoise_model.load_state_dict(checkpoint['state_dict'])
        print("Loaded denoise_model from checkpoint.")
    else:
        print("No denoiser checkpoint found. Model is uninitialized.")

import csv
from tqdm import tqdm

# Suppose we want to generate N graphs = len(test_loader).
# We'll assume each batch in test_loader is size 1 for simplicity,
# or you can group them in bigger batches if you wish.

model.eval()
denoise_model.eval()

with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["graph_id", "edge_list"])

    for k, data in enumerate(tqdm(test_loader, desc='Processing test set')):
        # Option A: unconditional sampling from noise
        #    We'll ignore the real data in `test_loader` except for storing filename
        # Option B: you might incorporate conditions from data.stats,
        #    in which case you'd pass them to the denoising model, etc.

        batch_size = data.num_graphs  # or data.stats.size(0), etc.
        # sample from the diffusion model
        #   shape [B, D_latent]
        x_sample = sample(
            denoise_model,
            batch_size=batch_size,
            D_latent=2 * args.latent_dim,  # global + local
            timesteps=args.timesteps,
            betas=betas,
            device=device
        )

        # now split x_sample into (z_global, z_local)
        Dg = args.latent_dim        # global
        Dl = args.latent_dim        # local
        z_global = x_sample[:, :Dg]
        z_local = x_sample[:, Dg:Dg+Dl]

        # decode adjacency
        adj_hat = model.decode(z_global, z_local)

        # build networkx graph or do however you want to parse edges
        # For instance:
        # (You could adapt your prior `construct_nx_from_adj()`)

        graph_ids = data.filename  # or however you track IDs
        for i in range(batch_size):
            A_i = adj_hat[i].detach().cpu().numpy()
            # convert adjacency to edge list
            # e.g.:
            edges_list = []
            N = A_i.shape[0]
            for r in range(N):
                for c in range(r+1, N):
                    if A_i[r, c] > 0.5:
                        edges_list.append((r, c))

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for (u,v) in edges_list])

            # Write row
            g_id = graph_ids[i] if hasattr(data, 'filename') else f"graph_{k}_{i}"
            writer.writerow([g_id, edge_list_text])



#########################################################################


"""
# define beta schedule
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

# initialize denoising model
denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_condition, d_cond=args.dim_condition).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# Train denoising model
if args.train_denoiser:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            loss.backward()
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()

        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'denoise_model.pth.tar')
else:
    checkpoint = torch.load('denoise_model.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])

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
        adj = autoencoder.decode_mu(x_sample)
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
            writer.writerow([graph_id, edge_list_text])"""