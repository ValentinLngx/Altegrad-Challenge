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



class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.05):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.projection = nn.Linear(in_dim, out_dim)  # Added projection layer for dimension matching
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        identity = self.projection(x)  # Project identity to match dimensions
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

        self.input_fc = nn.Linear(latent_dim + 7, hidden_dim)

        dims = [hidden_dim * (2 ** i) for i in range(n_layers - 1)]

        # Residual blocks with increasing sizes
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_dim=dims[i],
                hidden_dim=dims[i] * 2,  # Hidden dim is twice the input
                out_dim=dims[i + 1],  # Output dim matches next block's input
                dropout=dropout
            )
            for i in range(n_layers - 2)
        ])

        final_dim = hidden_dim * (2 ** (n_layers - 2))
        self.output_fc = nn.Linear(final_dim, 2 * n_nodes * (n_nodes - 1) // 2)

    def forward(self, x, stats):

        x = torch.cat((x, stats), dim=-1)             
        x = F.relu(self.input_fc(x))                   

        for block in self.res_blocks:
            x = block(x)                       

        x = self.output_fc(x)                       
        # Reshape to (batch_size, num_edges, 2)
        x = torch.reshape(x, (x.size(0), -1, 2))

        x = F.gumbel_softmax(x, tau=self.tau, hard=True)[:, :, 0]
        
        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)

        batch_size = adj.size(0)
        for b in range(batch_size):
            n_nodes = int(stats[b][0].item())
            adj[b, n_nodes:, :] = 0
            adj[b, :, n_nodes:] = 0

        return adj

    def update_temperature(self):
        """ Decays the Gumbel softmax temperature each epoch. """
        self.tau = max(self.final_tau, self.tau * self.tau_decay)


class AdjacencyCNNEncoder(nn.Module):
    """
    A convolutional encoder that takes as input a batch of adjacency matrices of 
    shape (B, 50, 50) and produces a latent vector of shape (B, latent_dim) for each graph.
    """
    def __init__(self, n_nodes=50, hidden_dim=256, out_dim=128):
        """
        Args:
            n_nodes: Number of nodes per graph (here 50).
            hidden_dim: Number of filters in the last conv layer.
            out_dim: The desired output dimension (should match the autoencoder's expected input,
                     e.g. hidden_dim_enc from your original GIN model).
        """
        super(AdjacencyCNNEncoder, self).__init__()
        # Input: (B, 1, 50, 50)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)    # -> (B, 32, 50, 50)
        self.bn1   = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # -> (B, 64, 25, 25)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # -> (B, 128, 13, 13)
        self.bn3   = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1) 
        # With hidden_dim filters, output: (B, hidden_dim, 7, 7)
        self.bn4   = nn.BatchNorm2d(hidden_dim)
        
        # Fully connected layer to map from flattened conv output to the desired latent vector.
        self.fc    = nn.Linear(hidden_dim * 7 * 7, out_dim)
        
    def forward(self, data):
        """
        Expects data.A to be the adjacency matrix of shape (B, 50, 50).
        """
        # Ensure data.A is a tensor.
        A = data.A
        if not isinstance(A, torch.Tensor):
            A = torch.as_tensor(A)
        A = A.float()  # cast to float
        
        # Add channel dimension: (B, 1, 50, 50)
        x = A.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))    # (B, 32, 50, 50)
        x = F.relu(self.bn2(self.conv2(x)))    # (B, 64, 25, 25)
        x = F.relu(self.bn3(self.conv3(x)))    # (B, 128, 13, 13)
        x = F.relu(self.bn4(self.conv4(x)))    # (B, hidden_dim, 7, 7)
        
        x = x.view(x.size(0), -1)              # (B, hidden_dim * 7 * 7)
        latent = self.fc(x)                    # (B, out_dim)
        return latent
    
    
class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        """
        Now input_dim should be 50 (the number of nodes) since each node’s feature is the corresponding
        row of the adjacency matrix.
        """
        super().__init__()
        self.dropout = dropout

        # Create a list of GIN convolution layers.
        self.convs = nn.ModuleList()
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
        """
        Instead of reading data.x and data.edge_index, we now assume that data.A is a dense
        adjacency matrix of shape (B, 50, 50) where B is the batch size.
        
        We will:
          1. Create node features by using each row of the adjacency as the feature for that node.
          2. Compute edge_index for each graph from nonzero entries of its adjacency.
          3. Build a batch vector so that global pooling knows how to separate graphs.
        """
        A = data.A  # shape: (B, 50, 50)
        B, n, _ = A.shape  # Here, n should be 50.
        
        # (1) Node features: each node’s feature = its row in A.
        # x will have shape (B*n, n). For each graph, we treat the i-th row of its 50x50 A as the feature for node i.
        x = A.view(B * n, n)
        
        # (2) Compute edge_index for each graph.
        edge_index_list = []
        for i in range(B):
            # For graph i, get nonzero entries (edge indices) from A[i].
            # Since A is binary, we set a threshold > 0 to determine edges.
            nonzero = (A[i] > 0).nonzero(as_tuple=False).T  # shape: (2, num_edges_in_graph)
            # Shift node indices for graph i by i * n (since nodes are stacked).
            nonzero = nonzero + i * n
            edge_index_list.append(nonzero)
        edge_index = torch.cat(edge_index_list, dim=1)  # shape: (2, total_edges)
        
        # (3) Create batch vector: nodes 0..n-1 belong to graph 0, nodes n..2n-1 to graph 1, etc.
        batch = torch.arange(B, device=A.device).unsqueeze(1).repeat(1, n).view(-1)
        
        # Pass through the GIN layers.
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Global pooling: sum of node embeddings per graph.
        out = global_add_pool(x, batch)  # (B, hidden_dim)
        out = self.bn(out)
        out = self.fc(out)
        return out


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        #self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.encoder = AdjacencyCNNEncoder(hidden_dim_enc, latent_dim)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = FastDecoder(
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

    def loss_function(self, data, stats, beta=0.005):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, stats)

        # Still using L1, but you might consider BCEWithLogitsLoss if your decoder outputs logits.
        recon = F.l1_loss(adj, data.A, reduction="mean")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta * kld

        return loss, recon, kld
