import os
import math
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import community as community_louvain
from transformers import AutoTokenizer, AutoModel

from torch import Tensor
from torch.utils.data import Dataset

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm
import scipy.sparse as sparse
from torch_geometric.data import Data

from extract_feats import extract_feats, extract_numbers


class TextEncoder:
    def __init__(self, model_name="bert-base-uncased", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

    def encode(self, text):
        # Tokenize and encode the text
        inputs = self.tokenizer(text,
                                return_tensors="pt",
                                max_length=self.max_length,
                                padding=True,
                                truncation=True)

        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as the sentence embedding
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings


def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim, text_encoder=None):
    if text_encoder is None:
        text_encoder = TextEncoder()

    data_lst = []
    if dataset == 'test':
        filename = './data/dataset_' + dataset + '.pt'
        desc_file = './data/' + dataset + '/test.txt'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)

                # Get both numerical features and text embeddings
                feats_stats = extract_numbers(desc)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                # Generate text embeddings
                text_embedding = text_encoder.encode(desc)

                data_lst.append(Data(
                    stats=feats_stats,
                    text_embedding=text_embedding,
                    filename=graph_id
                ))
            fr.close()
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')

    else:
        filename = './data/dataset_' + dataset + '.pt'
        graph_path = './data/' + dataset + '/graph'
        desc_path = './data/' + dataset + '/description'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            files = [f for f in os.listdir(graph_path)]
            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx + 1:]
                fread = os.path.join(graph_path, fileread)
                fstats = os.path.join(desc_path, filen + ".txt")

                # Graph processing (same as before)
                G = nx.read_graphml(fread) if extension == "graphml" else nx.read_edgelist(fread)
                G = nx.convert_node_labels_to_integers(G, ordering="sorted")

                # Rest of the graph processing code remains the same until...

                # Get graph embeddings
                adj, edge_index, x = process_graph(G, n_max_nodes, spectral_emb_dim)

                # Get both numerical features and text embeddings
                with open(fstats, 'r') as f:
                    desc = f.read().strip()

                feats_stats = extract_feats(fstats)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                # Generate text embeddings
                text_embedding = text_encoder.encode(desc)

                data_lst.append(Data(
                    x=x,
                    edge_index=edge_index,
                    A=adj,
                    stats=feats_stats,
                    text_embedding=text_embedding,
                    filename=filen
                ))

            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')

    return data_lst


def process_graph(G, n_max_nodes, spectral_emb_dim):
    """Helper function to process graph and get embeddings"""
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    node_list_bfs = []
    for ii in range(len(CGs)):
        node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
        degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)
        bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
        node_list_bfs += list(bfs_tree.nodes())

    adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
    adj = torch.from_numpy(adj_bfs).float()

    # Compute Laplacian eigenvectors
    diags = np.sum(adj_bfs, axis=0)
    diags = np.squeeze(np.asarray(diags))
    D = sparse.diags(diags).toarray()
    L = D - adj_bfs
    with np.errstate(divide="ignore"):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sparse.diags(diags).toarray()
    L = np.linalg.multi_dot((DH, L, DH))
    L = torch.from_numpy(L).float()
    eigval, eigvecs = torch.linalg.eigh(L)
    eigval = torch.real(eigval)
    eigvecs = torch.real(eigvecs)
    idx = torch.argsort(eigval)
    eigvecs = eigvecs[:, idx]

    edge_index = torch.nonzero(adj).t()

    # Create node features
    size_diff = n_max_nodes - G.number_of_nodes()
    x = torch.zeros(G.number_of_nodes(), spectral_emb_dim + 1)
    x[:, 0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:, 0] / (n_max_nodes - 1)
    mn = min(G.number_of_nodes(), spectral_emb_dim)
    mn += 1
    x[:, 1:mn] = eigvecs[:, :spectral_emb_dim]

    # Pad adjacency matrix
    adj = F.pad(adj, [0, size_diff, 0, size_diff])
    adj = adj.unsqueeze(0)

    return adj, edge_index, x

def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G



def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x




def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class CrossAttention(nn.Module):
    def __init__(self, graph_dim, text_dim, d_condition=128, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_condition // num_heads  # Using d_condition for consistent dimensionality
        self.scale = self.head_dim ** -0.5

        # Adjust dimensions to match projected text size (d_condition) and graph features
        self.to_q = nn.Linear(graph_dim, d_condition)
        self.to_k = nn.Linear(d_condition, d_condition)  # text is already projected to d_condition
        self.to_v = nn.Linear(d_condition, d_condition)

        self.to_out = nn.Sequential(
            nn.Linear(d_condition, graph_dim),  # Map back to graph dimension
            nn.Dropout(0.1)
        )

    def forward(self, graph_features, text_embedding):
        batch_size = graph_features.shape[0]

        # Reshape query, key, value
        q = self.to_q(graph_features)
        k = self.to_k(text_embedding)
        v = self.to_v(text_embedding)

        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention = torch.softmax(attention, dim=-1)

        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, -1, self.num_heads * self.head_dim)

        return self.to_out(out)




