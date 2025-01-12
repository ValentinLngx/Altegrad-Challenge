import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GATv2Conv, SAGEConv

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
    """
    Decoder that:
      1) Maps the (latent + stats) input into node embeddings of shape (batch_size, n_nodes, node_emb_dim)
      2) Computes adjacency logits by applying an MLP f(z_i, z_j) for each node pair
    """

    def __init__(self, latent_dim, stats_dim, node_emb_dim, edge_hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_nodes = n_nodes
        self.node_emb_dim = node_emb_dim

        # -------------------------
        # 1) Node-Embedding Generator
        # -------------------------
        # Simple approach:
        #   - Single or multi-layer MLP that outputs (n_nodes * node_emb_dim)
        #   - Then reshape into (batch_size, n_nodes, node_emb_dim)

        mlp_layers = []
        input_dim = latent_dim + stats_dim
        hidden_dim = edge_hidden_dim  # you can pick different dimension(s) here
        # Example: 2 hidden layers
        mlp_layers.append(nn.Linear(input_dim, hidden_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(hidden_dim, n_nodes * node_emb_dim))

        self.node_embedding_mlp = nn.Sequential(*mlp_layers)

        # -------------------------
        # 2) Edge Probability MLP
        # -------------------------
        # f(z_i, z_j) -> adjacency logit for edge (i,j)
        # We'll stack the node embeddings along dim=-1 and feed them into an MLP
        edge_mlp_layers = []
        # input size is 2*node_emb_dim because we cat(z_i, z_j)
        current_dim = 2 * node_emb_dim

        for _ in range(n_layers - 1):
            edge_mlp_layers.append(nn.Linear(current_dim, edge_hidden_dim))
            edge_mlp_layers.append(nn.ReLU())
            current_dim = edge_hidden_dim

        # Final layer outputs 1 logit for existence of edge
        edge_mlp_layers.append(nn.Linear(current_dim, 1))

        self.edge_mlp = nn.Sequential(*edge_mlp_layers)

    def forward(self, latent_z, stats):
        """
        Args:
            latent_z: (batch_size, latent_dim)
            stats:    (batch_size, stats_dim)
        Returns:
            adjacency_logits: (batch_size, n_nodes, n_nodes)
        """
        batch_size = latent_z.size(0)

        # -------------------------
        # 1) Generate Node Embeddings
        # -------------------------
        input_cat = torch.cat([latent_z, stats], dim=-1)  # shape: (B, latent_dim+stats_dim)
        node_emb = self.node_embedding_mlp(input_cat)  # shape: (B, n_nodes * node_emb_dim)
        node_emb = node_emb.view(batch_size, self.n_nodes, self.node_emb_dim)
        # node_emb[b, i] is the embedding for node i in batch sample b

        # -------------------------
        # 2) Compute Pairwise Edge Logits
        # -------------------------
        # adjacency_logits[b, i, j] = MLP([node_emb[b,i], node_emb[b,j]])
        adjacency_logits = torch.zeros(
            batch_size, self.n_nodes, self.n_nodes,
            device=node_emb.device
        )

        # A simple double-loop for clarity (fine for smaller n_nodes).
        # If n_nodes is large, you might vectorize or use a more advanced approach.
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                # cat node embeddings along dim=-1 => shape (B, 2*node_emb_dim)
                zij = torch.cat([node_emb[:, i, :], node_emb[:, j, :]], dim=-1)
                # pass through edge_mlp => shape (B, 1)
                logits_ij = self.edge_mlp(zij).squeeze(-1)  # shape (B,)

                # fill upper and lower triangle
                adjacency_logits[:, i, j] = logits_ij
                adjacency_logits[:, j, i] = logits_ij

        return adjacency_logits


"""class GIN(torch.nn.Module):
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
"""


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        # First SAGE layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))

        # Additional SAGE layers
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = global_add_pool(x, data.batch)
        x = self.bn(x)
        x = self.fc(x)
        return x


class GATEncoder(nn.Module):
    """
    Example GAT-based encoder with multiple attention heads,
    batch normalization, and a final linear layer to produce
    the 'latent' features.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, num_heads=1, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        # First GAT layer: from input_dim -> hidden_dim
        # heads: number of attention heads
        # each head has dimension hidden_dim // num_heads (must divide cleanly)
        self.convs.append(
            GATv2Conv(in_channels=input_dim, out_channels=hidden_dim // num_heads, heads=num_heads)
        )

        # Additional GAT layers
        for _ in range(n_layers - 1):
            self.convs.append(
                GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim // num_heads, heads=num_heads)
            )

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)  # Activation
            x = F.dropout(x, self.dropout, self.training)

        # Global pooling (summing node features per graph)
        x = global_add_pool(x, data.batch)

        # Batch normalization on pooled features
        x = self.bn(x)

        # Linear projection to latent_dim
        x = self.fc(x)
        return x


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, latent_dim, stats_dim, node_emb_dim, edge_hidden_dim, n_layers_enc,
                 n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GraphSAGEEncoder(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, stats_dim, node_emb_dim, edge_hidden_dim, n_layers_dec, n_max_nodes)

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
        z = self.reparameterize(mu, logvar)

        # Decoder now outputs adjacency logits, not adjacency directly
        adj_logits = self.decoder(z, stats)  # shape (B, n_nodes, n_nodes)

        # data.A should be the "ground-truth" adjacency matrix (0/1), same shape as adj_logits
        # If it's not the same shape, you might need to broadcast or reshape it
        recon_loss = F.binary_cross_entropy_with_logits(adj_logits, data.A, reduction="mean")

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kld

        return loss, recon_loss, kld

