import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        # Adjusting the input size of the first layer to account for the concatenated stats vector
        mlp_layers = [nn.Linear(latent_dim + 7, hidden_dim)] + [
            nn.Linear(hidden_dim*(2**i), hidden_dim*(2**(i+1))) for i in range(n_layers - 2)
        ]
        mlp_layers.append(nn.Linear(hidden_dim*2**(n_layers-2), n_nodes * (n_nodes - 1)))

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

        batch_size = adj.size(0)
        for b in range(batch_size):
            n_nodes = int(stats[b][0].item())  # Number of nodes for this batch
            adj[b, n_nodes:, :] = 0  # Mask rows beyond n_nodes
            adj[b, :, n_nodes:] = 0  # Mask columns beyond n_nodes

        return adj

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        
        # GIN layers with attention aggregation
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

        # Attention pooling for node embeddings
        self.attention_pool = nn.Linear(hidden_dim, 1)

        # Final layers
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.bn_global = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Attention pooling
        attention_weights = torch.sigmoid(self.attention_pool(x))
        out = global_add_pool(x * attention_weights, data.batch)
        
        # Batch norm and latent projection
        out = self.bn_global(out)
        out = self.fc_latent(out)
        
        return x, out  # Return node and graph-level representations


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)  # Updated encoder
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data, stats):
        # Encoder outputs node-level and graph-level representations
        x_nodes, x_g = self.encoder(data)

        # Graph-level latent variables
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)

        # Decoder processes the latent vector to reconstruct the graph
        adj = self.decoder(z, stats)
        return adj, mu, logvar

    def encode(self, data):
        x_nodes, x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=1.0):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar, stats):
        z = self.reparameterize(mu, logvar)
        adj = self.decoder(z, stats)
        return adj

    def decode_mu(self, mu, stats):
        adj = self.decoder(mu, stats)
        return adj

    def loss_function(self, data, stats, beta=0.005):
        # Encode graph
        x_nodes, x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)

        # Decode graph
        adj = self.decoder(z, stats)

        # Reconstruction loss
        recon = F.l1_loss(adj, data.A, reduction="mean")  # Replace `data.A` with your adjacency matrix representation

        # KLD regularization
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = recon + beta * kld
        return loss, recon, kld
