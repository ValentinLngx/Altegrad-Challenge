import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from torch_geometric.nn import GATConv, global_mean_pool

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, n_edges_enforce=None):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        
        probs = F.softmax(x, dim=-1)
        p_edge = probs[:, :, 1]

        if n_edges_enforce is not None:
            batch_size = p_edge.size(0)
            top_edges = []
            for b in range(batch_size):
                # Sort edges by probability (descending)
                sorted_vals, sorted_idx = torch.sort(p_edge[b], descending=True)
                mask = torch.zeros_like(p_edge[b])
                k = min(n_edges_enforce, p_edge.size(1))
                mask[sorted_idx[:k]] = 1.0
                top_edges.append(mask)
            x = torch.stack(top_edges, dim=0)
        else:
            # If we don't enforce edges, do the old Gumbel or Argmax logic
            x = F.gumbel_softmax(x, tau=1.0, hard=True)[:, :, 0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj




class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

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
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def loss_function(self, data, beta=0.05):
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld


class GlobalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_nodes, assign_ratio=0.25):
        """
        Example global encoder using DiffPool.
        - in_channels: dimension of node features.
        - hidden_dim: dimension of hidden layers.
        - num_nodes: maximum number of nodes in the graph (for DenseDiffPool).
        - assign_ratio: ratio of nodes to pool to.
        """
        super(GlobalEncoder, self).__init__()

        # Dense Convs for DiffPool
        self.gconv1 = DenseGCNConv(in_channels, hidden_dim)
        self.gconv2 = DenseGCNConv(hidden_dim, hidden_dim)

        # Assignment network
        self.assign_conv1 = DenseGCNConv(in_channels, hidden_dim)
        self.assign_conv2 = DenseGCNConv(hidden_dim, int(num_nodes * assign_ratio))

        self.hidden_dim = hidden_dim

    def forward(self, x, adj):
        """
        x: Dense node feature matrix [batch_size, num_nodes, in_channels]
        adj: Dense adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        # DiffPool requires dense adjacency, so be sure to use e.g. to_dense_batch if needed

        # Feature GCN forward pass
        s = F.relu(self.gconv1(x, adj))
        s = F.relu(self.gconv2(s, adj))

        # Assignment matrix forward pass
        a = F.relu(self.assign_conv1(x, adj))
        a = F.relu(self.assign_conv2(a, adj))

        # Now pool
        inter = dense_diff_pool(s, adj, a)
        x_pooled = inter[0]
        adj_pooled = inter[1]
        # x_pooled is [batch_size, num_clusters, hidden_dim]
        # For simplicity, do a mean pool across clusters -> single vector per graph
        x_global = x_pooled.mean(dim=1)  # shape [batch_size, hidden_dim]
        return x_global


class LocalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(LocalEncoder, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_dim, heads=1)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, batch):
        """
        x: node features in sparse format [num_nodes, in_channels]
        edge_index: [2, num_edges]
        batch: batch assignments for each node
        """
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))

        # Pool per-graph to get a single embedding (or a substructure embedding)
        x_local = global_mean_pool(x, batch)  # shape [num_graphs, hidden_dim]
        return x_local
