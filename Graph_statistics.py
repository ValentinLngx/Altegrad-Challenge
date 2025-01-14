#Graph statistics

import torch
import numpy as np
import networkx as nx
import community as community_louvain
import random, math, copy
import os
import csv
from torch import FloatTensor
from extract_feats import extract_numbers, extract_feats

random.seed(4200)
np.random.seed(4200)

def compute_graph_stats(adj):
    """
    Compute graph statistics from an adjacency matrix (undirected, no self-loops) or a NetworkX graph.
    Returns a vector of:
    [num_nodes, num_edges, average_degree, num_triangles, global_clustering, max_kcore, num_communities]
    """

    # Ensure adjacency is a NumPy array
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()

    #check if adj is a nx graph
    if isinstance(adj, nx.Graph):
        G = adj
    else:
        G = nx.from_numpy_array(adj)

    # Number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Average degree
    avg_degree = 2 * num_edges / num_nodes

    # Number of triangles
    triangles_per_node = nx.triangles(G)
    num_triangles = sum(triangles_per_node.values()) / 3

    # Global clustering coefficient
    all_triplets = sum(deg * (deg - 1) / 2 for _, deg in G.degree())
    global_clustering = (3 * num_triangles / all_triplets) if all_triplets > 0 else 0.0

    # Maximum k-core
    cores = nx.core_number(G)
    max_kcore = max(cores.values())

    # Number of communities (Louvain method)
    num_repeats = 20
    communities_list = [len(set(community_louvain.best_partition(G.copy()).values()))
                        for _ in range(num_repeats)]
    num_communities = np.mean(communities_list)

    return [
        int(num_nodes), 
        int(num_edges), 
        avg_degree, 
        int(num_triangles), 
        global_clustering, 
        max_kcore,
        int(num_communities)
    ]


def compute_graph_stats_from_edgelist(edgelist):
    """
    Compute graph statistics from an edge list.
    Returns a vector of:
    [num_nodes, num_edges, average_degree, num_triangles, global_clustering, max_kcore, num_communities]
    """
    G = nx.Graph()
    G.add_edges_from(edgelist)

    return compute_graph_stats(G)


def load_edgenodes_from_csv(file_path):
    """
    Load edge nodes from a CSV file, omitting the first line.
    
    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        list: A list of tuples, where each tuple contains graph ID and a list of edges.
    """
    edge_data = []
    
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        
        # Skip the first line
        next(reader)
        
        # Process the remaining lines
        for row in reader:
            graph_id = row[0]
            edge_str = row[1]
            
            # Parse edge list from string
            edges = eval(edge_str)  # Convert string "(0, 1), (0, 2)" to [(0, 1), (0, 2)]
            edge_data.append((graph_id, edges))
    
    return edge_data


def save_test_statistics():
    """
    Extracts and saves the statistics vectors from the test set description file.
    """
    stats_list = []

    # Read the test description file
    with open("./data/test/test.txt", "r") as fr:
        for line in fr:
            line = line.strip()
            tokens = line.split(",")
            #graph_id = tokens[0]
            desc = tokens[1:]
            desc = "".join(desc)  # Combine description tokens

            # Extract the statistics vector from the description
            feats_stats = extract_numbers(desc)
            #feats_stats = FloatTensor(feats_stats).unsqueeze(0)

            # Add to the list
            stats_list.append(feats_stats)

    return stats_list


from tqdm import tqdm

def save_train_statistics():
    """
    Extracts and saves the statistics vectors from the individual graph description files in the train set.
    """
    stats_list = []

    # Check if description folder exists
    desc_path = "./data/train/description"

    # Number of graphs to process (assumes files are named graph_0.txt to graph_7999.txt)
    num_graphs = 8000

    for i in range(num_graphs):
        file_name = f"graph_{i}.txt"
        file_path = os.path.join(desc_path, file_name)

        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        # Extract the statistics using the provided `extract_feats` function
        stats = extract_feats(file_path)
        stats_tensor = torch.FloatTensor(stats)

        # Append the statistics vector to the list
        stats_list.append(stats_tensor)

    return stats_list


def save_train_edgelists():
    """
    Saves the edgelists from the train set into a list.

    Returns:
        edgelist_list (list): A list containing the edgelists for all graphs.
    """
    edgelist_list = []
    graph_path = "./data/train/graph"

    # Ensure the path exists
    if not os.path.isdir(graph_path):
        raise FileNotFoundError(f"Graph path {graph_path} does not exist.")

    # Number of graphs to process (assumes files are named graph_0.edgelist to graph_7999.edgelist)
    num_graphs = 8000

    print(f"Processing {num_graphs} graph files...")

    for i in tqdm(range(num_graphs), desc="Loading graphs"):
        file_edgelist = f"graph_{i}.edgelist"
        file_graphml = f"graph_{i}.graphml"
        file_path_edgelist = os.path.join(graph_path, file_edgelist)
        file_path_graphml = os.path.join(graph_path, file_graphml)

        # Determine file type and load accordingly
        if os.path.isfile(file_path_edgelist):
            with open(file_path_edgelist, "r") as infile:
                edgelist = [tuple(map(int, line.split()[:2])) for line in infile]
                edgelist_list.append(edgelist)
        elif os.path.isfile(file_path_graphml):
            G = nx.read_graphml(file_path_graphml)
            edgelist = list(G.edges())
            edgelist_list.append(edgelist)
        else:
            raise FileNotFoundError(f"Neither .edgelist nor .graphml file found for graph_{i}.")

    print(f"Successfully loaded graphs for {len(edgelist_list)} graphs.")

    return edgelist_list


def z_normalize(data):
    """
    Perform z-normalization on a dataset.
    Z = (X - mean) / std
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return (data - mean) / std

def z_normalize_using_train(data, stat_train):
    """
    Perform z-normalization on a dataset.
    Z = (X - mean) / std
    """

    mean = np.mean(stat_train, axis=0)
    std = np.std(stat_train, axis=0)

    return (data - mean) / std



def calculate_mae(stats_pred, stats_real):
    """
    Calculate the MAE between two lists of vectors after z-normalization.
    
    Parameters:
        stats_pred (ndarray): Predicted statistics (shape: [n_samples, n_features]).
        stats_real (ndarray): Real statistics (shape: [n_samples, n_features]).
    
    Returns:
        float: The MAE score.
    """
    # Convert to NumPy arrays
    stats_pred = np.array(stats_pred)
    stats_real = np.array(stats_real)
    
    # Z-normalize both datasets
    stat_train = save_train_statistics()
    stats_pred_norm_using_train = z_normalize_using_train(stats_pred, stat_train)
    stats_real_norm_using_train = z_normalize_using_train(stats_real, stat_train)
    
    # Calculate the MAE
    #mae = np.mean(np.abs(stats_pred_norm_using_train - stats_real_norm_using_train))
    mean_abs_error_per_dimension = np.mean(np.abs(stats_pred_norm_using_train - stats_real_norm_using_train), axis=0)
    mae = np.mean(mean_abs_error_per_dimension)
    
    return mae

def get_score(file_path="output.csv"):

    """
    prints MAE score of the output.csv file and the test set
    """

    #compute and save stats vectors of output.csv
    #file_path = "output.csv"
    edges_lists = load_edgenodes_from_csv(file_path)

    stats_output = []
    for graph_id, edgelist in edges_lists:
        stats = compute_graph_stats_from_edgelist(edgelist)
        stats_output.append(stats)

    #compute and save stats vectors of testset
    stats_test = save_test_statistics()

    #compute MAE Score
    mae_score = calculate_mae(stats_output, stats_test)
    print("MAE Score:", mae_score)


def cost(G, target_stats, stat_train):
    """
    Compute a mae cost between the statistics of graph G
    and the target statistics vector.
    """
    current_stats = compute_graph_stats(G)
    stats_pred_norm_using_train = z_normalize_using_train(current_stats, stat_train)
    stats_real_norm_using_train = z_normalize_using_train(target_stats, stat_train)

    
    diff = stats_pred_norm_using_train - stats_real_norm_using_train
    mae = np.mean(np.abs(diff))
    return mae


def propose_modification(G):
    """
    Propose a new graph by modifying G either by a single edge flip or a block move
    (an edge swap between two pairs of nodes).
    """
    G_new = G.copy()
    # Decide randomly which move to try.
    if random.random() < 0.5:
        # --- Single edge flip ---
        nodes = list(G_new.nodes())
        if len(nodes) < 2:
            return G_new
        u, v = random.sample(nodes, 2)
        if G_new.has_edge(u, v):
            G_new.remove_edge(u, v)
        else:
            G_new.add_edge(u, v)
    else:
        # --- Block move: attempt an edge swap ---
        edges = list(G_new.edges())
        if len(edges) < 2:
            return G_new
        # Randomly select two edges
        e1, e2 = random.sample(edges, 2)
        u, v = e1
        x, y = e2
        
        # Avoid swapping if any nodes are repeated (to prevent self-loops or parallel issues)
        if len({u, v, x, y}) < 4:
            return G_new  # No operation if nodes are not distinct

        # Option 1: Swap to form edges (u, x) and (v, y)
        if not G_new.has_edge(u, x) and not G_new.has_edge(v, y) and u != x and v != y:
            G_new.remove_edge(u, v)
            G_new.remove_edge(x, y)
            G_new.add_edge(u, x)
            G_new.add_edge(v, y)
        # Option 2: Swap to form edges (u, y) and (v, x)
        elif not G_new.has_edge(u, y) and not G_new.has_edge(v, x) and u != y and v != x:
            G_new.remove_edge(u, v)
            G_new.remove_edge(x, y)
            G_new.add_edge(u, y)
            G_new.add_edge(v, x)
        # Else, if neither swap is applicable, the move is skipped (i.e. a no-op).
    return G_new



def refine_graph(initial_graph, target_stats, stat_train, steps=50, init_temp=1.0, cooling_rate=0.999):
    """
    Refine the graph to match target_stats using simulated annealing.
    
    Parameters:
      initial_graph: the starting NetworkX graph.
      target_stats: the target 7-dimensional statistics (as a NumPy array).
      steps: the number of iterations to run.
      init_temp: initial temperature for simulated annealing.
      cooling_rate: multiplicative cooling schedule (temperature *= cooling_rate each step).
      weights: optional vector of weights for cost components.
      
    Returns:
      The refined graph.
    """
    current_graph = initial_graph.copy()
    current_cost = cost(current_graph.copy(), target_stats, stat_train)
    best_graph = current_graph.copy()
    best_cost = current_cost
    temp = init_temp

    for i in range(steps):
        proposed_graph = propose_modification(current_graph.copy())
        proposed_cost = cost(proposed_graph, target_stats, stat_train)
        delta = proposed_cost - current_cost

        # Accept move if cost is reduced or with a probability if it increases
        #if delta < 0 or random.random() < np.exp(-delta / temp):
        if delta < 0:
            current_graph = proposed_graph
            current_cost = proposed_cost
            if current_cost < best_cost:
                best_graph = current_graph.copy()
                best_cost = current_cost
                #print("new best cost found: ", cost(proposed_graph, target_stats, stat_train))

        # Cool down the system.
        temp *= cooling_rate

        # Logging every 1000 iterations.
        #if i % 1000 == 0:
        #    print(f"Step {i}: current cost = {current_cost:.3f}, best cost = {best_cost:.3f}, temp = {temp:.3f}")

    return best_graph


def refine_graph():

    stat_train = save_train_statistics()

    file_path = "output.csv"
    edges_lists = load_edgenodes_from_csv(file_path)
    stats = save_test_statistics()

    refined_edgelists = []
    for i, edgelist in enumerate(edges_lists):
        
        G = nx.Graph()
        G.add_edges_from(edgelist[1])
        mae = cost(G, stats[i], stat_train)
        print("Initial Graph ", i, " MAE: ", mae)
        #print("Initial Graph Edges: ", list(G.edges()))

        refined_G = refine_graph(G.copy(), stats[i], stat_train)
        refined_edgelist = list(refined_G.edges())
        mae = cost(refined_G.copy(), stats[i], stat_train)
        print("Refined Graph ", i, " MAE: ", mae)
        #print("Refined Graph Edges: ", refined_edgelist)
        refined_edgelists.append(refined_edgelist)


    # Save refined edgelists to a CSV file
    with open("output2.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["graph_id", "edge_list"])
        
        for graph_id, edgelist in enumerate(tqdm(refined_edgelists, desc="Saving edgelists")):
            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in edgelist])
            # Write the graph ID and edge list to the CSV file
            writer.writerow([f"graph_{graph_id}", edge_list_text])


    get_score("output2.csv")

