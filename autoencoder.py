# autoencoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool


# ------------------------------------------------------------------
# 1. Decoder (no change here except using ReLU for consistency)
# ------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        # MLP layers
        mlp_layers = [nn.Linear(latent_dim, hidden_dim)]
        for _ in range(n_layers - 2):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        mlp_layers.append(nn.Linear(hidden_dim, n_nodes * (n_nodes - 1) // 2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()

    def forward(self, z_cond):
        x = z_cond
        for i in range(self.n_layers - 1):
            x = self.relu(self.mlp[i](x))
        logits = self.mlp[self.n_layers - 1](x)

        batch_size = logits.shape[0]
        adj = logits.new_zeros(batch_size, self.n_nodes, self.n_nodes)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)
        adj[:, idx[0], idx[1]] = logits
        adj = adj + adj.transpose(1, 2)
        return adj


# ------------------------------------------------------------------
# 2. GIN Encoder with simpler ReLU layers (NO BatchNorm)
# ------------------------------------------------------------------
class GIN(nn.Module):
    """
    GIN with skip connections. 
    Simplified to reduce risk of numerical issues.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        # Map input_dim -> hidden_dim so that skip connections match
        self.input_map = nn.Linear(input_dim, hidden_dim)

        # Build GIN layers, each stays in hidden_dim
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU()
                    )
                )
            )
        # Instead of BatchNorm, use Identity
        self.bn = nn.Identity()

        # Final linear to latent_dim
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x  # shape: [num_nodes, input_dim]

        # First map to hidden_dim
        x = self.input_map(x)

        # Apply each GIN layer with skip connections
        for conv in self.convs:
            x_skip = x
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, self.training)
            x = x + x_skip

        # Global pooling => shape: [batch_size, hidden_dim]
        out = global_add_pool(x, data.batch)

        # Identity instead of BN for safety
        out = self.bn(out)

        # Final linear
        out = self.fc(out)
        return out


# ------------------------------------------------------------------
# 3. Variational AutoEncoder with property conditioning + CLAMP
# ------------------------------------------------------------------
class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim_enc,
        hidden_dim_dec,
        latent_dim,
        n_layers_enc,
        n_layers_dec,
        n_max_nodes,
        n_condition=7,
        d_condition=128
    ):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes

        self.n_condition = n_condition
        self.d_condition = d_condition

        # small MLP to encode the property vector
        self.stats_encoder = nn.Sequential(
            nn.Linear(self.n_condition, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_condition),
            nn.ReLU(),
        )

        # GIN encoder
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)

        # Mu/LogVar from hidden_dim_enc + d_condition
        self.fc_mu = nn.Linear(hidden_dim_enc + self.d_condition, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc + self.d_condition, latent_dim)

        # Decoder: input is (latent_dim + d_condition)
        self.decoder = Decoder(latent_dim + self.d_condition, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def reparameterize(self, mu, logvar):
        # Clamp logvar to avoid large exponent
        logvar = torch.clamp(logvar, -10, 10)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def forward(self, data):
        x_g = self.encoder(data)                     # shape: [B, hidden_dim_enc]
        stats_embed = self.stats_encoder(data.stats) # shape: [B, d_condition]

        combined = torch.cat([x_g, stats_embed], dim=-1)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        z = self.reparameterize(mu, logvar)

        z_cond = torch.cat([z, stats_embed], dim=-1)
        adjacency_logits = self.decoder(z_cond)
        return adjacency_logits

    def encode(self, data):
        x_g = self.encoder(data)
        stats_embed = self.stats_encoder(data.stats)
        combined = torch.cat([x_g, stats_embed], dim=-1)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        z = self.reparameterize(mu, logvar)
        return z, stats_embed

    def decode_mu(self, z, stats=None):
        if stats is None:
            raise ValueError("Must provide 'stats' for conditioned decoding.")
        stats_embed = self.stats_encoder(stats)
        z_cond = torch.cat([z, stats_embed], dim=-1)
        return self.decoder(z_cond)

    def loss_function(self, data, beta=0.05, gamma=0.1):
        """
        data: includes:
          - A (adjacency) of shape [B, n_max_nodes, n_max_nodes]
          - stats (property vector), shape [B, n_condition]
        """
        x_g = self.encoder(data)
        stats_embed = self.stats_encoder(data.stats)
        combined = torch.cat([x_g, stats_embed], dim=-1)

        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        # Clamp logvar to reduce exploding variance
        logvar = torch.clamp(logvar, -10, 10)

        z = self.reparameterize(mu, logvar)

        # Decode => adjacency logits
        z_cond = torch.cat([z, stats_embed], dim=-1)
        adjacency_logits = self.decoder(z_cond)

        bsz, n_nodes, _ = adjacency_logits.shape
        target_adj = data.A.float()

        # 1) Reconstruction loss
        recon_loss = F.binary_cross_entropy_with_logits(
            adjacency_logits.view(bsz, -1),
            target_adj.view(bsz, -1),
            reduction='mean'
        )

        # 2) KL Divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld = kld / bsz

        # 3) Optional property matching
        property_loss = 0.
        if gamma > 0.:
            adj_hard = (torch.sigmoid(adjacency_logits) > 0.5).float()
            edges_pred = adj_hard.sum(dim=[1, 2]) / 2.0
            edges_true = data.stats[:, 1]  # assume stats[:,1] is #edges
            property_loss = F.mse_loss(edges_pred, edges_true)

        loss = recon_loss + beta * kld + gamma * property_loss
        return loss, recon_loss, kld
