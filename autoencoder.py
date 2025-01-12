import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Add projection layer if dimensions don't match
        self.projection = None
        if in_dim != hidden_dim:
            self.projection = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        identity = x

        # Apply projection if needed
        if self.projection is not None:
            identity = self.projection(x)

        # Main branch
        out = self.layer_norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)

        # Add residual connection
        return out + identity


class GraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.output_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # Apply layer normalization
        x = self.layer_norm(x)

        # Multi-head attention
        attended, _ = self.attention(x, x, x)

        # Process output
        out = self.output_layer(attended)
        return out

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        # Add layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(latent_dim + 7)

        # Use residual connections for better gradient flow
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                hidden_dim if i > 0 else latent_dim + 7,
                hidden_dim,
                dropout=0.1
            ) for i in range(n_layers - 1)
        ])

        # Output layer with attention mechanism
        self.attention = GraphAttention(hidden_dim, hidden_dim)
        self.final_layer = nn.Linear(hidden_dim, 2)

        # Temperature parameter for Gumbel-Softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, stats):
        # Concatenate and normalize inputs
        x = torch.cat((x, stats), dim=-1)
        x = self.layer_norm(x)

        # Generate node embeddings
        batch_size = x.size(0)
        node_embeddings = x.unsqueeze(1).expand(-1, self.n_nodes, -1)

        # Process through residual layers
        h = node_embeddings
        for layer in self.residual_layers:
            h = layer(h)

        # Apply attention mechanism
        h = self.attention(h)

        # Generate edge probabilities
        edge_logits = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                edge_input = torch.cat([h[:, i], h[:, j]], dim=-1)
                edge_logit = self.final_layer(edge_input)
                edge_logits.append(edge_logit)

        edge_logits = torch.stack(edge_logits, dim=1)

        # Apply Gumbel-Softmax with learned temperature
        x = F.gumbel_softmax(edge_logits, tau=self.temperature.abs(), hard=True)[:, :, 0]

        # Convert to adjacency matrix
        adj = torch.zeros(batch_size, self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)

        return adj



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
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim_dec,
            n_layers=n_layers_dec,
            n_nodes=n_max_nodes
        )

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

        recon = F.l1_loss(adj, data.A, reduction="mean")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta * kld

        return loss, recon, kld
