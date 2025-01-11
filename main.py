# main.py

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
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

# Our modules
from autoencoder import VariationalAutoEncoder  # This is the updated AE w/ clamp
from denoise_model import DenoiseNN, p_losses, sample, ImprovedDenoiseNN
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset

np.random.seed(13)



parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate
parser.add_argument('--lr', type=float, default=5e-5) 

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
parser.add_argument('--latent-dim', type=int, default=32)

# Max number of nodes
parser.add_argument('--n-max-nodes', type=int, default=50)

# Number of layers for encoder/decoder
parser.add_argument('--n-layers-encoder', type=int, default=2)
parser.add_argument('--n-layers-decoder', type=int, default=3)

# Spectral embedding dimension
parser.add_argument('--spectral-emb-dim', type=int, default=10)

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=100)

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

# --------------------------------------------------------------------------
# 1) Preprocess dataset
# --------------------------------------------------------------------------
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim)
testset  = preprocess_dataset("test",  args.n_max_nodes, args.spectral_emb_dim)

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader  = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# --------------------------------------------------------------------------
# 2) Initialize VAE
# --------------------------------------------------------------------------
autoencoder = VariationalAutoEncoder(
    input_dim=args.spectral_emb_dim + 1,
    hidden_dim_enc=args.hidden_dim_encoder,
    hidden_dim_dec=args.hidden_dim_decoder,
    latent_dim=args.latent_dim,
    n_layers_enc=args.n_layers_encoder,
    n_layers_dec=args.n_layers_decoder,
    n_max_nodes=args.n_max_nodes,
    n_condition=args.n_condition,
    d_condition=args.dim_condition
).to(device)

optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler_ae = torch.optim.lr_scheduler.StepLR(optimizer_ae, step_size=500, gamma=0.1)

# --------------------------------------------------------------------------
# 3) Train autoencoder (with gradient clipping)
# --------------------------------------------------------------------------
if args.train_autoencoder:
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs_autoencoder + 1):
        autoencoder.train()
        train_loss_all = 0.0
        train_recon_all = 0.0
        train_kld_all = 0.0
        cnt_train = 0

        for data in train_loader:
            data = data.to(device)
            optimizer_ae.zero_grad()
            # property matching is turned off if gamma=0
            loss, recon, kld = autoencoder.loss_function(data, 
                                                            beta=args.beta, 
                                                            gamma=args.gamma)
            loss.backward()
            
            # Gradient clipping to avoid exploding grads
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=5.0)
            optimizer_ae.step()
            
            train_loss_all   += loss.item()
            train_recon_all  += recon.item()
            train_kld_all    += kld.item()
            cnt_train        += 1

        # Validation
        autoencoder.eval()
        val_loss_all  = 0.0
        val_recon_all = 0.0
        val_kld_all   = 0.0
        cnt_val       = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                v_loss, v_recon, v_kld = autoencoder.loss_function(
                    data, beta=args.beta, gamma=args.gamma
                )
                val_loss_all  += v_loss.item()
                val_recon_all += v_recon.item()
                val_kld_all   += v_kld.item()
                cnt_val       += 1

        # Logging
        dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_t} Epoch: {epoch:03d}, '
                f'Train Loss: {train_loss_all/cnt_train:.5f}, '
                f'Train Recon: {train_recon_all/cnt_train:.5f}, '
                f'Train KLD: {train_kld_all/cnt_train:.5f}, '
                f'Val Loss: {val_loss_all/cnt_val:.5f}, '
                f'Val Recon: {val_recon_all/cnt_val:.5f}, '
                f'Val KLD: {val_kld_all/cnt_val:.5f}')

        scheduler_ae.step()

        # Save best
        if (val_loss_all < best_val_loss):
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer': optimizer_ae.state_dict(),
            }, 'autoencoder.pth.tar')
else:
    # Load from checkpoint
    checkpoint = torch.load('autoencoder.pth.tar', map_location=device)
    autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()

# --------------------------------------------------------------------------
# 4) Prepare diffusion model
# --------------------------------------------------------------------------
betas = linear_beta_schedule(timesteps=args.timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

denoise_model = ImprovedDenoiseNN(
    input_dim=args.latent_dim,
    hidden_dim=args.hidden_dim_denoise,
    n_layers=args.n_layers_denoise,
    n_cond=args.n_condition,
    d_cond=args.dim_condition
).to(device)

optimizer_denoise = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler_denoise = torch.optim.lr_scheduler.StepLR(optimizer_denoise, step_size=500, gamma=0.1)

# --------------------------------------------------------------------------
# 5) Train the denoiser (with gradient clipping)
# --------------------------------------------------------------------------
if args.train_denoiser:
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs_denoise + 1):
        denoise_model.train()
        train_loss_all = 0.0
        train_count = 0

        for data in train_loader:
            data = data.to(device)
            optimizer_denoise.zero_grad()

            z, _ = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (z.size(0),), device=device).long()
            loss = p_losses(
                denoise_model,
                z,
                t, 
                data.stats,
                sqrt_alphas_cumprod, 
                sqrt_one_minus_alphas_cumprod,
                loss_type="huber"
            )
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in denoise_model.named_parameters() if "attention" in n], 
                max_norm=1.0
            )
            optimizer_denoise.step()

            train_loss_all += z.size(0) * loss.item()
            train_count    += z.size(0)

        # Validation
        denoise_model.eval()
        val_loss_all = 0.0
        val_count = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                z, _ = autoencoder.encode(data)
                t = torch.randint(0, args.timesteps, (z.size(0),), device=device).long()
                v_loss = p_losses(
                    denoise_model, z, t, data.stats,
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                    loss_type="huber"
                )
                val_loss_all += z.size(0) * v_loss.item()
                val_count    += z.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f'{dt_t} Epoch: {epoch:03d}, '
                    f'Train Loss: {train_loss_all/train_count:.5f}, '
                    f'Val Loss: {val_loss_all/val_count:.5f}')

        scheduler_denoise.step()

        # Save best
        if val_loss_all < best_val_loss:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer': optimizer_denoise.state_dict(),
            }, 'denoise_model.pth.tar')
else:
    # Load from checkpoint
    checkpoint = torch.load('denoise_model.pth.tar', map_location=device)
    denoise_model.load_state_dict(checkpoint['state_dict'])

denoise_model.eval()

# Done training, free up memory
del train_loader, val_loader

# --------------------------------------------------------------------------
# 6) Inference / Generate on Test set
# --------------------------------------------------------------------------
with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["graph_id", "edge_list"])

    for k, data in enumerate(tqdm(test_loader, desc='Processing test set')):
        data = data.to(device)
        stat = data.stats
        bs = stat.size(0)

        graph_ids = data.filename

        # Sample from diffusion
        samples = sample(
            denoise_model,
            stat,
            latent_dim=args.latent_dim,
            timesteps=args.timesteps,
            betas=betas,
            batch_size=bs
        )
        z_sample = samples[-1]

        # Decode => must pass stats
        adj_logits = autoencoder.decode_mu(z_sample, stats=stat)
        adj_prob   = torch.sigmoid(adj_logits)
        adj_bin    = (adj_prob > 0.5).float()

        for i in range(bs):
            G_generated = construct_nx_from_adj(adj_bin[i].detach().cpu().numpy())
            graph_id = graph_ids[i]

            edge_list_text = ", ".join([f"({u}, {v})" for u, v in G_generated.edges()])
            writer.writerow([graph_id, edge_list_text])

