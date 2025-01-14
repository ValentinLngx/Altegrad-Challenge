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

class VirtualNode(nn.Module):
    """
    Manages a learnable 'virtual node' embedding that is updated
    by aggregating representations of real nodes at each layer
    and redistributing them back to the nodes.
    """
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        # One embedding for each layer or a single embedding reused
        self.embedding = nn.Parameter(torch.zeros(n_layers, hidden_dim))
        # Initialize to small random values
        nn.init.xavier_uniform_(self.embedding.view(n_layers, 1, hidden_dim))

        # MLP used to update the virtual node after aggregation
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, batch, layer_idx):
        """
        x: (N, hidden_dim) Node embeddings
        batch: (N,) Batch index
        layer_idx: which GNN layer we are in
        """
        # Aggregate from real nodes
        vn = self.embedding[layer_idx].unsqueeze(0)  # (1, hidden_dim)
        # Summation from nodes -> single vector per graph
        vn_batch = global_add_pool(x, batch)  # (batch_size, hidden_dim)
        # We assume a single graph or we do it per graph if multiple graphs in a batch
        # For multiple graphs, we'd repeat the virtual node for each graph
        # but for simplicity, we treat single-graph scenario or do the same for each subgraph

        # Update the virtual node
        vn_updated = self.mlp[layer_idx](vn_batch)

        # Redistribute the virtual node info back to the real nodes
        x = x + vn_updated[batch]

        return x, vn_updated


########################################################################
# Optional: Attention-based Jumping Knowledge
########################################################################
class JumpingKnowledgeAttn(nn.Module):
    """
    Performs an attention-based combination of multiple
    layer representations [h1, h2, ..., hL].
    """
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        self.attn_weights = nn.Parameter(torch.zeros(n_layers, hidden_dim))
        nn.init.xavier_uniform_(self.attn_weights)

        # An additional projection layer for each representation
        self.proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, hidden_states):
        """
        hidden_states: list of (N, hidden_dim)
        Returns: (N, hidden_dim)
        """
        # We learn a separate attention key for each layer
        # Then combine all layers in a data-dependent way
        # For simplicity, do a scalar attention per layer
        # or do a bigger transform if you want.
        # We'll do a simple dot-product-based approach:
        #    alpha_l = softmax(  h_l . W_l  )  across layers
        # Then we weigh each h_l by alpha_l

        # Stack hidden states: (L, N, hidden_dim)
        stacked = torch.stack(hidden_states, dim=0)

        # Weighted sums per layer
        # shape of self.attn_weights: (L, hidden_dim)
        # shape of stacked: (L, N, hidden_dim)
        # do a small transform before attention
        proj_list = []
        for i, (h, proj) in enumerate(zip(stacked, self.proj)):
            proj_list.append(proj(h))  # (N, hidden_dim)
        proj_stacked = torch.stack(proj_list, dim=0)  # (L, N, hidden_dim)

        # attention logits: (L, N)
        logits = (proj_stacked * self.attn_weights[:, None, :]).sum(dim=-1)
        # alpha: (L, N)
        alpha = F.softmax(logits, dim=0)

        # Weighted sum
        out = (proj_stacked * alpha.unsqueeze(-1)).sum(dim=0)  # (N, hidden_dim)
        return out


########################################################################
# Enhanced MLP for GINConv
########################################################################
class GatedResidualMLP(nn.Module):
    """
    A Gated MLP that can serve as the update function in GIN.
    Incorporates LayerNorm, dropout, and a residual gated connection.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.gate = nn.Linear(in_dim, out_dim)  # gating for residual
        self.dropout = dropout

    def forward(self, x):
        residual = x
        # Gate for the residual connection
        gate = torch.sigmoid(self.gate(residual))

        # MLP
        out = F.gelu(self.norm1(self.fc1(x)))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.gelu(self.norm2(self.fc2(out)))
        out = F.dropout(out, p=self.dropout, training=self.training)

        # Gated residual
        out = gate * out + (1 - gate) * residual
        return out


########################################################################
# Main Enhanced GIN Model
########################################################################
class EnhancedGIN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        n_layers,
        dropout=0.2,
        virtual_node=True,
        aggregator_type="sum",  # "sum", "mean", "max"...
        jk_mode="attention",    # "attention" or "weighted_sum" or "last"
        use_vae=False,          # If True, outputs mu and logvar for a VAE
    ):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.virtual_node = virtual_node
        self.jk_mode = jk_mode
        self.use_vae = use_vae

        # Feature normalization
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Define the aggregator function for GINConv
        # By default GIN is "sum" aggregator. PyG's GINConv also allows "mean" or "max".
        # We'll just do standard GINConv but you can adapt aggregator_type if needed
        # in a custom aggregator. For demonstration, we keep it simple with GINConv.
        def build_conv(in_dim, out_dim):
            mlp = GatedResidualMLP(in_dim, out_dim, hidden_dim, dropout)
            # GINConv aggregator is by default sum-based, but you can pass
            # aggregator_type='mean' or 'max' by customizing a custom message passing.
            return GINConv(nn=mlp, train_eps=True)

        # Convs and norms
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(build_conv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.convs.append(build_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Optional Virtual Node
        if virtual_node:
            self.virtualnode = VirtualNode(hidden_dim, n_layers)

        # Jumping Knowledge
        if jk_mode == "attention":
            self.jump = JumpingKnowledgeAttn(hidden_dim, n_layers)
        elif jk_mode == "weighted_sum":
            self.jump_weights = nn.Parameter(torch.ones(n_layers))
        else:
            # 'last' or other simpler modes
            self.jump = None

        # Multi-head readout
        self.pool_weight = nn.Parameter(torch.ones(3))

        # Final projection: either deterministic or to (mu, logvar) for VAE
        out_dim = 2 * latent_dim if use_vae else latent_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial BN
        x = self.input_bn(x)

        hidden_states = []

        for layer_idx, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # Optional Virtual Node update (before or after conv, can experiment)
            if self.virtual_node:
                x, vn_embedding = self.virtualnode(x, batch, layer_idx)

            x = conv(x, edge_index)
            x = bn(x)
            x = F.gelu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            hidden_states.append(x)

        # Jumping Knowledge
        if self.jk_mode == "attention":
            x = self.jump(hidden_states)  # (N, hidden_dim)
        elif self.jk_mode == "weighted_sum":
            weights = F.softmax(self.jump_weights, dim=0)
            x = sum(w * h for w, h in zip(weights, hidden_states))
        else:
            # 'last' layer only
            x = hidden_states[-1]

        # Multi-head pooling across the final node embeddings
        pool_weights = F.softmax(self.pool_weight, dim=0)
        pooled = (
            global_add_pool(x, batch) * pool_weights[0] +
            global_mean_pool(x, batch) * pool_weights[1] +
            global_max_pool(x, batch) * pool_weights[2]
        )

        # Final MLP projection
        out = self.fc1(pooled)
        out = self.layer_norm(out)
        out = F.gelu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        # residual
        out = out + pooled
        out = self.fc2(out)

        # If VAE mode, split into mu and logvar
        if self.use_vae:
            mu, logvar = torch.split(out, out.size(-1)//2, dim=-1)
            return mu, logvar  # You can then do the reparameterization
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
