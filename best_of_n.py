


import torch
import csv
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
import argparse

#from main import parser
from autoencoder import VariationalAutoEncoder
from denoise_model import sample, ImprovedDenoiseNN
from utils import construct_nx_from_adj, preprocess_dataset, get_diffusion_parameters
from Graph_statistics import save_train_statistics, save_test_statistics
from utils import preprocess_dataset, construct_nx_from_adj
from postprocessing import cost_batch


parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate
parser.add_argument('--lr', type=float, default=1e-3)

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.0)

# Batch size for training
parser.add_argument('--batch-size', type=int, default=250)

# Number of epochs for autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=100)

# Hidden dimension sizes
parser.add_argument('--hidden-dim-encoder', type=int, default=128)
parser.add_argument('--hidden-dim-decoder', type=int, default=128)

# Latent dimension
parser.add_argument('--latent-dim', type=int, default=128)

# Max number of nodes
parser.add_argument('--n-max-nodes', type=int, default=50)

# Number of layers for encoder/decoder
parser.add_argument('--n-layers-encoder', type=int, default=2)
parser.add_argument('--n-layers-decoder', type=int, default=6)

# Spectral embedding dimension
parser.add_argument('--spectral-emb-dim', type=int, default=20)

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=30)

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=500)

# Hidden dimension for denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512)

# Number of layers for denoising model
parser.add_argument('--n-layers_denoise', type=int, default=3)

# Flags for training
parser.add_argument('--train-autoencoder', action='store_false', default=True)
parser.add_argument('--train-denoiser', action='store_true', default=True)

# Conditioning dimension
parser.add_argument('--dim-condition', type=int, default=128)
parser.add_argument('--n-condition', type=int, default=7)

# Optional: clamp gamma to 0 if you suspect property loss is causing NaNs
parser.add_argument('--gamma', type=float, default=0.0, 
                    help="Weight for property matching loss; set to 0 at first to avoid NaNs")

# Beta for KL term
parser.add_argument('--beta', type=float, default=0.05, 
                    help="KL term weight in the VAE loss")


args = parser.parse_args()







stats_train = save_train_statistics()
stats_test = save_test_statistics()

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

autoencoder = VariationalAutoEncoder(50, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)
checkpoint = torch.load('autoencoder.pth.tar')
autoencoder.load_state_dict(checkpoint['state_dict'])
autoencoder.eval()

denoise_model = ImprovedDenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_condition, d_cond=args.dim_condition).to(device)
checkpoint = torch.load('denoise_model.pth.tar')
denoise_model.load_state_dict(checkpoint['state_dict'])
denoise_model.eval()

diff_params = get_diffusion_parameters(args.timesteps, beta_schedule="cosine")
betas = diff_params['betas']


if __name__ == '__main__':
    with open("output2.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["graph_id", "edge_list"])

        # Process each batch in the test loader
        for k, data in enumerate(tqdm(test_loader, desc='Processing test set')):
            data = data.to(device)
            stat = data.stats
            bs = stat.size(0)
            graph_ids = data.filename

            # Containers to record the best candidate for each graph in the batch
            best_costs = [float('inf')] * bs
            best_adjs = [None] * bs

            stat_d = torch.reshape(stat, (-1, args.n_condition))

            # For each candidate, generate a sample and update the best candidate if its cost is lower
            for candidate in range(20):
                samples = sample(
                    denoise_model, 
                    data.stats, 
                    latent_dim=args.latent_dim, 
                    timesteps=args.timesteps, 
                    betas=betas, 
                    batch_size=bs
                )
                x_sample = samples[-1]
                candidate_adj_tensor = autoencoder.decode_mu(x_sample, stat)
                candidate_adj = candidate_adj_tensor.detach().cpu().numpy()

                batch_target_stats = stats_test[k * bs : k * bs + bs]
                candidate_costs = cost_batch(
                    candidate_adj, 
                    target_stats=batch_target_stats, 
                    stat_train=stats_train, 
                    n_jobs=12
                )

                # Update best candidates using the batch cost values
                for i in range(bs):
                    if candidate_costs[i] < best_costs[i]:
                        best_costs[i] = candidate_costs[i]
                        best_adjs[i] = candidate_adj[i]

            # Write the best candidate for each graph to the CSV file
            for i in range(bs):
                Gs_generated = construct_nx_from_adj(best_adjs[i])
                graph_id = graph_ids[i]

                edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])
                writer.writerow([graph_id, edge_list_text])

    #from Graph_statistics import get_score
    #get_score("output5.csv")

