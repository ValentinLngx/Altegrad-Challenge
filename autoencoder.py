import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool

# Decoder
"""class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        # Adjusting the input size of the first layer to account for the concatenated stats vector
        mlp_layers = [nn.Linear(latent_dim + 7, hidden_dim)] + [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)
        ]
        mlp_layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, stats):
        # Concatenate the stats vector to the latent representation
        x = torch.cat((x, stats), dim=-1)

        # Pass through the MLP layers
        for i in range(self.n_layers - 1):
            x = self.relu(self.mlp[i](x))

        x = self.mlp[self.n_layers - 1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:, :, 0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj
"""

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        # Adjusting the input size of the first layer to account for the concatenated stats vector
        mlp_layers = [nn.Linear(latent_dim + 7, hidden_dim)] + [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)
        ]
        mlp_layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, stats):
        # Concatenate the stats vector to the latent representation
        x = torch.cat((x, stats), dim=-1)

        # Pass through the MLP layers
        for i in range(self.n_layers - 1):
            x = self.silu(self.mlp[i](x))

        x = self.mlp[self.n_layers - 1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:, :, 0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.05):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        # x has shape: (batch_size, in_dim)
        identity = x
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.activation(out + identity)  # skip connection + activation


class FastDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, dropout=0.1, initial_tau=2.0, final_tau=0.5, tau_decay=0.995):
        super(FastDecoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        self.tau = initial_tau
        self.final_tau = final_tau
        self.tau_decay = tau_decay
        # First layer: from (latent_dim + 7) to hidden_dim
        # so dimension changes from (latent_dim + 7) -> hidden_dim
        self.input_fc = nn.Linear(latent_dim + 7, hidden_dim)

        # Residual blocks (each keeps dimension == hidden_dim)
        # We'll do n_layers - 2 residual blocks to match your original structure
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_layers - 2)
        ])

        # Final layer: from hidden_dim -> 2 * n_nodes*(n_nodes - 1)//2
        self.output_fc = nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2)

    def forward(self, x, stats):
        # Concatenate the stats vector to the latent representation
        x = torch.cat((x, stats), dim=-1)  # shape: (batch_size, latent_dim+7)

        # First layer + activation
        x = F.silu(self.input_fc(x))  # shape: (batch_size, hidden_dim)

        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x)  # still (batch_size, hidden_dim)

        # Final linear layer (no activation here, since we do gumbel_softmax next)
        x = self.output_fc(x)  # shape: (batch_size, 2*num_edges)

        # Reshape to (batch_size, num_edges, 2)
        x = torch.reshape(x, (x.size(0), -1, 2))

        # Gumbel softmax
        x = F.gumbel_softmax(x, tau=self.tau, hard=True)[:, :, 0] # shape: (batch_size, num_edges)

        # Build adjacency matrix
        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)

        return adj

    def update_temperature(self):
        """ Decays the Gumbel softmax temperature each epoch. """
        self.tau = max(self.final_tau, self.tau * self.tau_decay)

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                )
            )
        )
        for _ in range(n_layers - 1):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LeakyReLU(0.2),
                        nn.BatchNorm1d(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LeakyReLU(0.2),
                    )
                )
            )

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = FastDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim_dec,
            n_layers=n_layers_dec,
            n_nodes=n_max_nodes
        )
        # Add cyclic KL annealing parameters
        self.kl_cycle_length = 20  # Length of each cycle
        self.current_epoch = 0

    def get_kl_weight(self):
        # Implements cyclic KL annealing
        if self.training:
            cycle_position = (self.current_epoch % self.kl_cycle_length) / self.kl_cycle_length
            if cycle_position < 0.5:
                # Increasing phase
                kl_weight = cycle_position * 2
            else:
                # Constant phase
                kl_weight = 1.0
            return kl_weight
        return 1.0  # During evaluation, use full KL term

    def forward(self, data, stats):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, stats)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.0):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar, stats):
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, stats)
        return adj

    def decode_mu(self, mu, stats):
        adj = self.decoder(mu, stats)
        return adj

    def loss_function(self, data, stats, beta=0.05):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, stats)

        # Reconstruction loss with edge importance weighting
        edge_weights = torch.ones_like(data.A)
        edge_weights = edge_weights + (data.A * 2.0)  # Weight existing edges more heavily
        recon = F.binary_cross_entropy_with_logits(adj, data.A, weight=edge_weights, reduction="mean")
        #recon = F.l1_loss(adj, data.A, reduction="mean")

        # KL divergence with annealing
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weight = self.get_kl_weight()

        # Structural loss using graph properties
        structural_loss = self.structural_loss(adj, data.A)

        # Combined loss with weighted terms
        loss = recon + kl_weight * kld + 0.1 * structural_loss
        #loss = recon + beta * kld

        return loss, recon, kld

    def structural_loss(self, pred_adj, true_adj):
        # Add structural properties matching
        pred_degree = torch.sum(pred_adj, dim=2)
        true_degree = torch.sum(true_adj, dim=2)
        degree_loss = F.mse_loss(pred_degree, true_degree)

        # Add clustering coefficient approximation
        pred_triangles = torch.bmm(torch.bmm(pred_adj, pred_adj), pred_adj)
        true_triangles = torch.bmm(torch.bmm(true_adj, true_adj), true_adj)
        clustering_loss = F.mse_loss(pred_triangles, true_triangles)

        return degree_loss + clustering_loss
