import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

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

from typing import List, Tuple


class ResidualBlock(nn.Module):
    """Improved residual block with layer normalization and better initialization."""

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            dropout: float = 0.05,
            use_layer_norm: bool = True
    ):
        super().__init__()

        # Layer normalization for better training stability
        self.layer_norm1 = nn.LayerNorm(in_dim) if use_layer_norm else nn.Identity()
        self.layer_norm2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        # Main network layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.projection = nn.Linear(in_dim, out_dim)

        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

        # Initialize weights using Kaiming initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.projection.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-layer normalization architecture.

        Args:
            x: Input tensor of shape (batch_size, in_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim)
        """
        # Identity branch with projection
        identity = self.projection(x)

        # Main branch with layer norm, activation, and dropout
        out = self.layer_norm1(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.layer_norm2(out)
        out = self.fc2(out)

        return self.activation(out + identity)


class FastDecoder(nn.Module):
    """
    Improved FastDecoder with better architecture and temperature scheduling.

    Args:
        latent_dim: Dimension of the latent space
        hidden_dim: Initial hidden dimension
        n_layers: Number of layers in the network
        n_nodes: Number of nodes in the output graph
        dropout: Dropout probability
        initial_tau: Initial temperature for Gumbel-Softmax
        final_tau: Final temperature for Gumbel-Softmax
        tau_decay: Temperature decay rate
    """

    def __init__(
            self,
            latent_dim: int,
            hidden_dim: int,
            n_layers: int,
            n_nodes: int,
            dropout: float = 0.1,
            initial_tau: float = 2.0,
            final_tau: float = 0.5,
            tau_decay: float = 0.995
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.stats_dim = 7  # Making this explicit as a class attribute

        # Temperature parameters for Gumbel-Softmax
        self.register_buffer('tau', torch.tensor(initial_tau))
        self.register_buffer('final_tau', torch.tensor(final_tau))
        self.tau_decay = tau_decay

        # Input processing
        self.input_norm = nn.LayerNorm(latent_dim + self.stats_dim)
        self.input_fc = nn.Linear(latent_dim + self.stats_dim, hidden_dim)

        # Create progressive growing architecture
        self.res_blocks = self._build_residual_blocks(hidden_dim, n_layers, dropout)

        # Output processing
        final_dim = hidden_dim * (2 ** (n_layers - 2))
        self.output_dim = 2 * n_nodes * (n_nodes - 1) // 2
        self.output_fc = nn.Linear(final_dim, self.output_dim)

        # Initialize weights
        self._initialize_weights()

    def _build_residual_blocks(self, hidden_dim: int, n_layers: int, dropout: float) -> nn.ModuleList:
        """Constructs the residual blocks with progressive growing architecture."""
        dims = [hidden_dim * (2 ** i) for i in range(n_layers - 1)]

        return nn.ModuleList([
            ResidualBlock(
                in_dim=dims[i],
                hidden_dim=dims[i] * 2,
                out_dim=dims[i + 1],
                dropout=dropout
            )
            for i in range(n_layers - 2)
        ])

    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        nn.init.kaiming_normal_(self.input_fc.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.output_fc.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x: Latent representation tensor of shape (batch_size, latent_dim)
            stats: Statistics tensor of shape (batch_size, 7)

        Returns:
            torch.Tensor: Adjacency matrix of shape (batch_size, n_nodes, n_nodes)
        """
        # Combine input and apply normalization
        x = torch.cat((x, stats), dim=-1)
        x = self.input_norm(x)
        x = F.silu(self.input_fc(x))

        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Generate adjacency matrix
        x = self.output_fc(x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=self.tau, hard=True)[:, :, 0]

        # Convert to symmetric adjacency matrix
        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + adj.transpose(1, 2)

        return adj

    @torch.no_grad()
    def update_temperature(self):
        """Updates the Gumbel-Softmax temperature using exponential decay."""
        self.tau = torch.maximum(
            self.final_tau,
            self.tau * self.tau_decay
        )

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
"""
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.input_bn = nn.BatchNorm1d(input_dim)

        # MLP with skip connections and layer normalization
        def enhanced_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )

        # Trainable epsilon for each GIN layer
        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_layers)
        ])

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GINConv(enhanced_mlp(input_dim, hidden_dim), train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.convs.append(GINConv(enhanced_mlp(hidden_dim, hidden_dim), train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Multi-head pooling
        self.pool_weight = nn.Parameter(torch.ones(3))

        # Output layers with residual connections
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Jump connections
        self.jump = nn.Parameter(torch.ones(n_layers))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial normalization
        x = self.input_bn(x)

        # Store all intermediate representations
        hidden_states = []

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.dropout(x, self.dropout, training=self.training)
            hidden_states.append(x)

        # Jump connections (weighted sum of all layers)
        jump_weights = F.softmax(self.jump, dim=0)
        x = sum(h * w for h, w in zip(hidden_states, jump_weights))

        # Multi-head pooling
        pool_weights = F.softmax(self.pool_weight, dim=0)
        pooled = (
                global_add_pool(x, batch) * pool_weights[0] +
                global_mean_pool(x, batch) * pool_weights[1] +
                global_max_pool(x, batch) * pool_weights[2]
        )

        # Output with residual connection
        out = self.fc1(pooled)
        out = self.layer_norm(out)
        out = F.gelu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = out + pooled  # Residual connection
        out = self.fc2(out)

        return out"""
class GatedResidualMLP(nn.Module):
    """
    Example MLP: linear -> BN -> GeLU -> dropout -> linear
    Outputs dimension out_dim. You can add gating/residuals as desired.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x_in = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


##############################################################################
# Example JumpingKnowledgeAttn (make sure it outputs shape [N, hidden_dim]).
##############################################################################
class JumpingKnowledgeAttn(nn.Module):
    """
    A simple attention-based JK aggregator that outputs [N, hidden_dim].
    """
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Learnable attention parameters (example)
        self.att_weight = nn.Parameter(torch.randn(num_layers, hidden_dim))

    def forward(self, hidden_states):
        """
        hidden_states: list of length num_layers,
                       each shape [N, hidden_dim].
        Output shape: [N, hidden_dim]
        """
        # Stack all states => shape [num_layers, N, hidden_dim]
        x_all = torch.stack(hidden_states, dim=0)

        # For each layer, compute an attention score = dot with att_weight
        # att_weight => [num_layers, hidden_dim]
        # x_all       => [num_layers, N, hidden_dim]
        # => scores   => [num_layers, N]
        scores = torch.einsum('lhd, lhd -> l h', x_all, self.att_weight.unsqueeze(1))
        # Actually, you'd need to adapt the above. This is just a toy example.
        # Typically you'd do something like:
        #   score_layer = (x_all[l] * self.att_weight[l]).sum(dim=-1)
        #   ...
        # We'll just do a naive softmax across layers.
        scores = torch.mean(x_all * self.att_weight.unsqueeze(1), dim=-1)  # [num_layers, N]
        alpha = F.softmax(scores, dim=0)  # [num_layers, N]

        # Weighted sum of hidden_states by alpha => [N, hidden_dim]
        # shape of alpha is [num_layers, N], so broadcast multiply
        x = (x_all * alpha.unsqueeze(-1)).sum(dim=0)  # [N, hidden_dim]
        return x


##############################################################################
# Example VirtualNode (make sure it returns (x, vn_embedding) consistently).
##############################################################################
class VirtualNode(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vnode_proj = nn.Linear(input_dim, hidden_dim)  # Fix input size here

    def forward(self, x, batch, layer_idx):
        vn_embedding = global_mean_pool(x, batch)  # Now shape [batch_size, input_dim]
        vn_embedding = self.vnode_proj(vn_embedding)  # Now correctly maps to [batch_size, hidden_dim]
        x = x + vn_embedding[batch]  # Add back virtual node embedding
        return x, vn_embedding


##############################################################################
# The EnhancedGIN Model
##############################################################################
class EnhancedGIN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        n_layers_enc,
        dropout=0.2,
        virtual_node=False,
        aggregator_type="sum",  # "sum", "mean", "max"...
        jk_mode="attention",    # "attention" or "weighted_sum" or "last"
        use_vae=False,          # If True, outputs mu and logvar for a VAE
    ):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers_enc
        self.virtual_node = virtual_node
        self.jk_mode = jk_mode
        self.use_vae = use_vae
        self.hidden_dim = hidden_dim

        # Feature normalization
        self.input_bn = nn.BatchNorm1d(input_dim)

        def build_conv(in_dim, out_dim):
            mlp = GatedResidualMLP(in_dim, out_dim, hidden_dim, dropout)
            # GINConv aggregator defaults to sum-based;
            # to fully respect aggregator_type, you'd need a custom message passing.
            return GINConv(nn=mlp, train_eps=True)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer: (input_dim -> hidden_dim)
        self.convs.append(build_conv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers: (hidden_dim -> hidden_dim)
        for _ in range(n_layers - 1):
            self.convs.append(build_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        if virtual_node:
            self.virtualnode = VirtualNode(hidden_dim, n_layers)
        else:
            self.virtualnode = None

        ######################################################################
        # Jumping Knowledge
        ######################################################################
        if jk_mode == "attention":
            # Make sure your JumpingKnowledgeAttn returns [N, hidden_dim]
            self.jump = JumpingKnowledgeAttn(hidden_dim, n_layers)
        elif jk_mode == "weighted_sum":
            # Weighted sum across layers
            self.jump_weights = nn.Parameter(torch.ones(n_layers))
            self.jump = None
        else:
            # 'last' or other simpler modes
            self.jump = None

        self.pool_weight = nn.Parameter(torch.ones(3))

        ######################################################################
        # Final projection: either deterministic or (mu, logvar) for VAE
        ######################################################################
        out_dim = 2 * latent_dim if use_vae else latent_dim
        # We do a small MLP:
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial BN
        x = self.input_bn(x)

        hidden_states = []

        for layer_idx, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # (Optional) Virtual Node update
            if self.virtual_node:
                x, vn_embedding = self.virtualnode(x, batch, layer_idx)

            x = conv(x, edge_index)
            x = bn(x)
            x = F.gelu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            hidden_states.append(x)

        ######################################################################
        # Jumping Knowledge
        ######################################################################
        if self.jk_mode == "attention":
            x = self.jump(hidden_states)  # shape [N, hidden_dim]
        elif self.jk_mode == "weighted_sum":
            # Weighted sum across hidden states
            weights = F.softmax(self.jump_weights, dim=0)  # shape [n_layers]
            x = sum(w * h for w, h in zip(weights, hidden_states))  # [N, hidden_dim]
        else:
            # 'last' layer only
            x = hidden_states[-1]

        ######################################################################
        # Multi-head pooling across the final node embeddings
        ######################################################################
        pool_weights = F.softmax(self.pool_weight, dim=0)  # shape [3]
        pooled = (
            global_add_pool(x, batch) * pool_weights[0] +
            global_mean_pool(x, batch) * pool_weights[1] +
            global_max_pool(x, batch) * pool_weights[2]
        )
        # pooled => shape [batch_size, hidden_dim]

        ######################################################################
        # Final MLP projection (with a small residual on the first layer)
        ######################################################################
        out = self.fc1(pooled)               # [batch_size, hidden_dim]
        out = self.layer_norm(out)           # [batch_size, hidden_dim]
        out = F.gelu(out)                    # [batch_size, hidden_dim]
        out = F.dropout(out, self.dropout, training=self.training)
        out = out + pooled                   # Residual with the pooled embedding
        out = self.fc2(out)                  # [batch_size, out_dim]

        # If VAE mode, split into mu and logvar
        if self.use_vae:
            mu, logvar = torch.split(out, out.size(-1)//2, dim=-1)
            return mu, logvar
        else:
            return out


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = EnhancedGIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
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
