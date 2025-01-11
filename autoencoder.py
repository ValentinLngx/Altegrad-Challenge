# autoencoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool
from utils import TextEncoder, CrossAttention


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
            d_condition=128,
            text_dim=768  # BERT's default output dimension
    ):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.n_condition = n_condition
        self.d_condition = d_condition

        # Text encoder for the description
        self.text_encoder = TextEncoder()

        # Project text embedding first (before cross-attention)
        self.text_projection = nn.Linear(text_dim, self.d_condition)

        # Cross-attention layer with correct dimensions
        self.cross_attention = CrossAttention(
            graph_dim=hidden_dim_enc,
            text_dim=self.d_condition,  # Use projected text_dim
            d_condition=self.d_condition
        )

        # Original components
        self.stats_encoder = nn.Sequential(
            nn.Linear(self.n_condition, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_condition),
            nn.ReLU(),
        )

        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)

        # Modified to account for text features
        # Combined dimensions: hidden_dim_enc + d_condition + text_dim
        self.fc_mu = nn.Linear(hidden_dim_enc + self.d_condition + text_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc + self.d_condition + text_dim, latent_dim)

        # Decoder now takes additional text features
        self.decoder = Decoder(latent_dim + self.d_condition + text_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        # Get graph features
        x_g = self.encoder(data)  # shape: [B, hidden_dim_enc]

        # Get text embedding
        text_desc = data.text_embedding  # Ensure data has 'text_desc' attribute
        text_embedding = self.text_encoder.encode(text_desc)  # shape: [B, text_dim]

        # Project text embedding before cross-attention
        text_emb_proj = self.text_projection(text_embedding)  # shape: [B, d_condition]

        # Apply cross-attention using projected text embedding
        enhanced_features = self.cross_attention(
            x_g.unsqueeze(1),  # shape: [B, 1, hidden_dim_enc]
            text_emb_proj.unsqueeze(1)  # shape: [B, 1, d_condition]
        ).squeeze(1)  # shape: [B, hidden_dim_enc]

        # Get stats embedding
        stats_embed = self.stats_encoder(data.stats)  # shape: [B, d_condition]

        # Combine all features: [hidden_dim_enc] + [d_condition] + [text_dim]
        combined = torch.cat([enhanced_features, stats_embed, text_embedding],
                             dim=-1)  # [B, hidden_dim_enc + d_condition + text_dim]

        # Continue with VAE logic
        mu = self.fc_mu(combined)  # [B, latent_dim]
        logvar = self.fc_logvar(combined)  # [B, latent_dim]
        z = self.reparameterize(mu, logvar)  # [B, latent_dim]

        # Include text embedding in decoder conditioning
        z_cond = torch.cat([z, stats_embed, text_embedding], dim=-1)  # [B, latent_dim + d_condition + text_dim]
        adjacency_logits = self.decoder(z_cond)  # Decoder output

        return adjacency_logits

    def encode(self, data):
        x_g = self.encoder(data)  # [B, hidden_dim_enc]
        print(data)
        text_desc = data.text_embedding
        text_embedding = self.text_encoder.encode(text_desc)  # [B, text_dim]

        # Project text embedding before cross-attention
        text_emb_proj = self.text_projection(text_embedding)  # [B, d_condition]

        # Apply cross-attention using projected text embedding
        enhanced_features = self.cross_attention(
            x_g.unsqueeze(1),  # [B, 1, hidden_dim_enc]
            text_emb_proj.unsqueeze(1)  # [B, 1, d_condition]
        ).squeeze(1)  # [B, hidden_dim_enc]

        stats_embed = self.stats_encoder(data.stats)  # [B, d_condition]

        # Combine all features
        combined = torch.cat([enhanced_features, stats_embed, text_embedding],
                             dim=-1)  # [B, hidden_dim_enc + d_condition + text_dim]

        mu = self.fc_mu(combined)  # [B, latent_dim]
        logvar = self.fc_logvar(combined)  # [B, latent_dim]
        z = self.reparameterize(mu, logvar)  # [B, latent_dim]

        return z, stats_embed, text_embedding

    def decode_mu(self, z, stats=None, text_desc=None):
        if stats is None or text_desc is None:
            raise ValueError("Must provide both 'stats' and 'text_desc' for conditioned decoding.")

        stats_embed = self.stats_encoder(stats)  # [B, d_condition]
        text_embedding = self.text_encoder.encode(text_desc)  # [B, text_dim]

        # Project text embedding before cross-attention
        text_emb_proj = self.text_projection(text_embedding)  # [B, d_condition]

        # Since decode_mu doesn't use cross-attention, but follows similar pattern
        # Here, assuming decoder needs the same conditioning as in forward
        # Combine conditioning features
        z_cond = torch.cat([z, stats_embed, text_embedding], dim=-1)  # [B, latent_dim + d_condition + text_dim]
        adjacency_logits = self.decoder(z_cond)

        return adjacency_logits

    def loss_function(self, data, beta=0.05, gamma=0.1):
        x_g = self.encoder(data)  # [B, hidden_dim_enc]

        # Handle text embedding
        if hasattr(data, 'text_desc'):
            text_embedding = self.text_encoder.encode(data.text_embedding)  # [B, text_dim]
            text_emb_proj = self.text_projection(text_embedding)  # [B, d_condition]
        else:
            # If no text description, use zero vector for projected text embedding
            text_embedding = torch.zeros(x_g.size(0), 768).to(x_g.device)  # Assuming text_dim=768
            text_emb_proj = torch.zeros(x_g.size(0), self.d_condition).to(x_g.device)

        # Apply cross-attention using projected text embedding
        enhanced_features = self.cross_attention(
            x_g.unsqueeze(1),  # [B, 1, hidden_dim_enc]
            text_emb_proj.unsqueeze(1)  # [B, 1, d_condition]
        ).squeeze(1)  # [B, hidden_dim_enc]

        # Get stats embedding
        stats_embed = self.stats_encoder(data.stats)  # [B, d_condition]

        # Combine features: [enhanced_features] + [stats_embed] + [raw text_embedding]
        combined = torch.cat([enhanced_features, stats_embed, text_embedding],
                             dim=-1)  # [B, hidden_dim_enc + d_condition + text_dim]

        mu = self.fc_mu(combined)  # [B, latent_dim]
        logvar = self.fc_logvar(combined)  # [B, latent_dim]
        logvar = torch.clamp(logvar, -10, 10)  # Stability
        z = self.reparameterize(mu, logvar)  # [B, latent_dim]

        # Include text embedding in decoder conditioning
        z_cond = torch.cat([z, stats_embed, text_embedding], dim=-1)  # [B, latent_dim + d_condition + text_dim]
        adjacency_logits = self.decoder(z_cond)  # [B, n_max_nodes, n_max_nodes]

        bsz, n_nodes, _ = adjacency_logits.shape
        target_adj = data.A.float()  # [B, n_max_nodes, n_max_nodes]

        # Reconstruction loss
        recon_loss = F.binary_cross_entropy_with_logits(
            adjacency_logits.view(bsz, -1),
            target_adj.view(bsz, -1),
            reduction='mean'
        )

        # KL Divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld = kld / bsz

        # Property matching loss
        property_loss = 0.
        if gamma > 0.:
            adj_hard = (torch.sigmoid(adjacency_logits) > 0.5).float()  # [B, n_max_nodes, n_max_nodes]
            edges_pred = adj_hard.sum(dim=[1, 2]) / 2.0  # [B]
            edges_true = data.stats[:, 1]  # Assuming second stat is number of edges
            property_loss = F.mse_loss(edges_pred, edges_true)

        loss = recon_loss + beta * kld + gamma * property_loss
        return loss, recon_loss, kld