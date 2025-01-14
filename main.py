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
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample, ImprovedDenoiseNN, test_gaussian_properties, q_sample
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset, get_diffusion_parameters

scaler = StandardScaler()
from torch.utils.data import Subset
np.random.seed(13)

"""
Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate
parser.add_argument('--lr', type=float, default=1e-3)

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.0)

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256)

# Number of epochs for autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=200)

# Hidden dimension sizes
parser.add_argument('--hidden-dim-encoder', type=int, default=64)
parser.add_argument('--hidden-dim-decoder', type=int, default=256)

# Latent dimension
parser.add_argument('--latent-dim', type=int, default=64)

# Max number of nodes
parser.add_argument('--n-max-nodes', type=int, default=50)

# Number of layers for encoder/decoder
parser.add_argument('--n-layers-encoder', type=int, default=4)
parser.add_argument('--n-layers-decoder', type=int, default=6)

# Spectral embedding dimension
parser.add_argument('--spectral-emb-dim', type=int, default=10)

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=10)

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=500)

# Hidden dimension for denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512)

# Number of layers for denoising model
parser.add_argument('--n-layers_denoise', type=int, default=3)

# Flags for training
parser.add_argument('--train-autoencoder', action='store_false', default=True)
parser.add_argument('--train-denoiser', action='store_true', default=True)

# Conditioning dimension
parser.add_argument('--dim-condition', type=int, default=128)
parser.add_argument('--n-condition', type=int, default=7)

# Optional: clamp gamma to 0 if you suspect property loss is causing NaNs
parser.add_argument('--gamma', type=float, default=0.0, 
                    help="Weight for property matching loss; set to 0 at first to avoid NaNs")

# Beta for KL term
parser.add_argument('--beta', type=float, default=0.05, 
                    help="KL term weight in the VAE loss")


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

print('OK')
# initialize VGAE model
autoencoder = VariationalAutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_autoencoder, eta_min=1e-6)


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
            loss, recon, kld  = autoencoder.loss_function(data, data.stats)
            train_loss_all_recon += recon.item()
            train_loss_all_kld += kld.item()
            cnt_train+=1
            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch)+1
            optimizer.step()
        autoencoder.decoder.update_temperature()
        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_all_recon = 0
        val_loss_all_kld = 0

        for data in val_loader:
            data = data.to(device)
            loss, recon, kld  = autoencoder.loss_function(data, data.stats)
            val_loss_all_recon += recon.item()
            val_loss_all_kld += kld.item()
            val_loss_all += loss.item()
            cnt_val+=1
            val_count += torch.max(data.batch)+1

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f},Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(dt_t,epoch, train_loss_all/cnt_train, train_loss_all_recon/cnt_train, train_loss_all_kld/cnt_train, val_loss_all/cnt_val, val_loss_all_recon/cnt_val, val_loss_all_kld/cnt_val))
            
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

autoencoder.eval()


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

diff_params = get_diffusion_parameters(args.timesteps, beta_schedule="cosine")
# Move all tensors to GPU
betas = diff_params['betas'].to(device)
alphas = diff_params['alphas'].to(device)
alphas_cumprod = diff_params['alphas_cumprod'].to(device)
alphas_cumprod_prev = diff_params['alphas_cumprod_prev'].to(device)
sqrt_recip_alphas = diff_params['sqrt_recip_alphas'].to(device)
sqrt_alphas_cumprod = diff_params['sqrt_alphas_cumprod'].to(device)
sqrt_one_minus_alphas_cumprod = diff_params['sqrt_one_minus_alphas_cumprod'].to(device)
posterior_variance = diff_params['posterior_variance'].to(device)

# Initialize denoising model
denoise_model = ImprovedDenoiseNN(
    input_dim=args.latent_dim,
    hidden_dim=args.hidden_dim_denoise,
    n_layers=args.n_layers_denoise,
    n_cond=args.n_condition,
    d_cond=args.dim_condition
).to(device)  # This is already correct in your code

optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# Training loop modifications
if args.train_denoiser:
    best_val_loss = float('inf')
    current_timesteps = args.timesteps

    for epoch in range(1, args.epochs_denoise + 1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0

        # Recalculate diffusion parameters with current_timesteps
        diff_params = get_diffusion_parameters(current_timesteps, beta_schedule="cosine")
        # Move all new tensors to GPU
        betas = diff_params['betas'].to(device)
        alphas = diff_params['alphas'].to(device)
        alphas_cumprod = diff_params['alphas_cumprod'].to(device)
        alphas_cumprod_prev = diff_params['alphas_cumprod_prev'].to(device)
        sqrt_recip_alphas = diff_params['sqrt_recip_alphas'].to(device)
        sqrt_alphas_cumprod = diff_params['sqrt_alphas_cumprod'].to(device)
        sqrt_one_minus_alphas_cumprod = diff_params['sqrt_one_minus_alphas_cumprod'].to(device)
        posterior_variance = diff_params['posterior_variance'].to(device)

        for data in train_loader:
            # Make sure data is moved to GPU
            data = data.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Enable automatic mixed precision
                x_g = autoencoder.encode(data)
                t_final = torch.full((x_g.size(0),), current_timesteps - 1, device=device, dtype=torch.long)
                noise = torch.randn_like(x_g)
                z_T = q_sample(x_g, t_final, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)

                # Test Gaussian properties
                test_results = test_gaussian_properties(z_T)

                if (not test_results['is_gaussian'] or test_results[
                    'confidence_score'] < 0.2) and current_timesteps < 20000:
                    current_timesteps = min(current_timesteps + 1000, 20000)
                    #print(f"\nEpoch {epoch}: Increasing timesteps to {current_timesteps}")
                    #print(f"Gaussian test confidence: {test_results['confidence_score']:.4f}")

                    # Recalculate and move to GPU
                    diff_params = get_diffusion_parameters(current_timesteps, beta_schedule="cosine")
                    for key in diff_params:
                        diff_params[key] = diff_params[key].to(device)

                    # Update local variables
                    locals().update(diff_params)

                t = torch.randint(0, current_timesteps, (x_g.size(0),), device=device).long()
                loss = p_losses(
                    denoise_model,
                    x_g,
                    t,
                    data.stats,
                    sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod,
                    loss_type="huber"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in denoise_model.named_parameters() if "attention" in n],
                max_norm=1.0
            )
            optimizer.step()

            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)

        # Validation loop
        denoise_model.eval()
        val_loss_all = 0
        val_count = 0

        with torch.no_grad():  # Add no_grad context for validation
            for data in val_loader:
                data = data.to(device)
                with torch.cuda.amp.autocast():  # Enable automatic mixed precision
                    x_g = autoencoder.encode(data)
                    t = torch.randint(0, current_timesteps, (x_g.size(0),), device=device).long()
                    loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod,
                                    sqrt_one_minus_alphas_cumprod, loss_type="huber")
                val_loss_all += x_g.size(0) * loss.item()
                val_count += x_g.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(
                dt_t, epoch, train_loss_all / train_count, val_loss_all / val_count))
            # Add GPU memory usage monitoring
            print(
                f'GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB allocated, {torch.cuda.memory_reserved() / 1024 ** 2:.1f}MB reserved')

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'denoise_model.pth.tar')
else:
    checkpoint = torch.load('denoise_model.pth.tar', map_location=device)  # Add map_location
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
        adj = autoencoder.decode_mu(x_sample, stat)
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