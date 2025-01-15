import random
import numpy as np
import networkx as nx
import csv
from tqdm import tqdm
import concurrent.futures
from Graph_statistics import save_train_statistics, save_test_statistics, load_edgenodes_from_csv
from Graph_statistics import get_score, compute_graph_stats, z_normalize_using_train


def compute_graph_stats_parallel(adj, batch=False, n_jobs=-1):
    """
    Compute graph statistics either for a single graph or a batch of graphs.
    If batch=True, then `adj` should be an iterable of adjacency matrices or NetworkX graphs.
    Uses parallel processing over the graphs with n_jobs workers.
    """
    if not batch:
        return compute_graph_stats(adj)
    else:
        # Process a list of adjacency matrices / graphs in parallel.
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(compute_graph_stats, adj))
        return results


def cost(G, target_stats, stat_train, return_stats=False):
    """
    Compute a mae cost between the statistics of graph G
    and the target statistics vector.
    """
    current_stats = compute_graph_stats(G)
    stats_pred_norm_using_train = z_normalize_using_train(current_stats, stat_train)
    stats_real_norm_using_train = z_normalize_using_train(target_stats, stat_train)

    
    diff = stats_pred_norm_using_train - stats_real_norm_using_train
    mae = np.mean(np.abs(diff))
    if return_stats:
        return mae, current_stats
    return mae


def cost_batch(G_list, target_stats, stat_train, n_jobs=4):
    """
    Compute the cost for each graph in a batch (list G_list) in parallel.
    Returns a list of MAE values.
    """
    # Compute statistics for each graph in parallel
    stats_list = compute_graph_stats_parallel(G_list, batch=True, n_jobs=n_jobs)
    # Normalize predictions and target
    normalized_stats = z_normalize_using_train(np.array(stats_list), stat_train)
    normalized_target = z_normalize_using_train(np.array(target_stats), stat_train)
    
    # Compute the MAE for each graph
    mae_list = np.mean(np.abs(normalized_stats - normalized_target), axis=1)
    return mae_list


def propose_modification(G):
    
    #Propose a new graph by modifying G either by a single edge flip or a block move
    #(an edge swap between two pairs of nodes).

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
"""

def propose_modification(G, current_stats, target_stats):
    
    #Propose a new graph by modifying G using current and target graph statistics.
    
    #Parameters:
    #  G: the current NetworkX graph.
    #  current_stats: a sequence with the current graph statistics as 
    #                 [num_nodes, num_edges, average_degree, num_triangles, global_clustering, max_kcore, num_communities].
    #  target_stats: a sequence with the target graph statistics (same order as above).
    #  
    #Returns:
    #  A new graph (copy of G) with one modification.
    
    
    G_new = G.copy()
    
    # If target_stats (or current_stats) is not provided, fall back to the original random approach.
    if target_stats is None or current_stats is None:
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
            # --- Block move ---
            edges = list(G_new.edges())
            if len(edges) < 2:
                return G_new
            e1, e2 = random.sample(edges, 2)
            u, v = e1
            x, y = e2
            if len({u, v, x, y}) < 4:
                return G_new
            if not G_new.has_edge(u, x) and not G_new.has_edge(v, y) and u != x and v != y:
                G_new.remove_edge(u, v)
                G_new.remove_edge(x, y)
                G_new.add_edge(u, x)
                G_new.add_edge(v, y)
            elif not G_new.has_edge(u, y) and not G_new.has_edge(v, x) and u != y and v != x:
                G_new.remove_edge(u, v)
                G_new.remove_edge(x, y)
                G_new.add_edge(u, y)
                G_new.add_edge(v, x)
        return G_new
    
    current_num_edges = current_stats[1]
    current_clustering = current_stats[4]
    target_num_edges = target_stats[1]
    target_clustering = target_stats[4]
    
    # Determine if we want to add edges more frequently.
    if current_num_edges < target_num_edges:
        add_edge_prob = 0.7
    else:
        add_edge_prob = 0.3

    # Decide whether to try an addition move or a removal/block move.
    if random.random() < add_edge_prob:
        # --- Edge Addition ---
        # If clustering is below target, try completing triangles first.
        if current_clustering < target_clustering and random.random() < 0.6:
            nodes = list(G_new.nodes())
            if not nodes:
                return G_new
            u = random.choice(nodes)
            neighbors = list(G_new.neighbors(u))
            candidate = None
            # Look for two neighbors that are not connected.
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if not G_new.has_edge(neighbors[i], neighbors[j]):
                        candidate = (neighbors[i], neighbors[j])
                        break
                if candidate is not None:
                    break
            if candidate is not None:
                G_new.add_edge(*candidate)
                return G_new
        # Fallback: try a simple addition.
        nodes = list(G_new.nodes())
        if len(nodes) < 2:
            return G_new
        u, v = random.sample(nodes, 2)
        if not G_new.has_edge(u, v):
            G_new.add_edge(u, v)
        return G_new
    else:
        # --- Edge Removal or Block Move ---
        # If clustering is higher than target, try removing an edge from a triangle.
        if current_clustering > target_clustering and random.random() < 0.5:
            edges = list(G_new.edges())
            if not edges:
                return G_new
            candidate = None
            for u, v in edges:
                # If there are common neighbors, the edge is likely part of a triangle.
                if len(list(nx.common_neighbors(G_new, u, v))) > 0:
                    candidate = (u, v)
                    break
            if candidate is not None:
                G_new.remove_edge(*candidate)
            else:
                # Fallback: remove any random edge.
                u, v = random.choice(edges)
                G_new.remove_edge(u, v)
            return G_new
        else:
            # Otherwise, try a block move (edge swap).
            edges = list(G_new.edges())
            if len(edges) < 2:
                return G_new
            e1, e2 = random.sample(edges, 2)
            u, v = e1
            x, y = e2
            # Ensure nodes are distinct.
            if len({u, v, x, y}) < 4:
                return G_new
            if not G_new.has_edge(u, x) and not G_new.has_edge(v, y) and u != x and v != y:
                G_new.remove_edge(u, v)
                G_new.remove_edge(x, y)
                G_new.add_edge(u, x)
                G_new.add_edge(v, y)
            elif not G_new.has_edge(u, y) and not G_new.has_edge(v, x) and u != y and v != x:
                G_new.remove_edge(u, v)
                G_new.remove_edge(x, y)
                G_new.add_edge(u, y)
                G_new.add_edge(v, x)
            return G_new
"""

def refine_graph(initial_graph, target_stats, stat_train, steps=100, init_temp=1.0, cooling_rate=0.999):
    """
    Refine the graph to match target_stats using simulated annealing.
    
    Parameters:
      initial_graph: the starting NetworkX graph.
      target_stats: the target 7-dimensional statistics (as a NumPy array).
      steps: the number of iterations to run.
      init_temp: initial temperature for simulated annealing.
      cooling_rate: multiplicative cooling schedule (temperature *= cooling_rate each step).
      
    Returns:
      The refined graph.
    """
    current_graph = initial_graph.copy()
    current_cost, current_stats = cost(current_graph.copy(), target_stats, stat_train, return_stats=True)

    if current_cost == 0:
        return current_graph
    
    best_graph = current_graph.copy()
    best_cost = current_cost
    temp = init_temp

    for _ in range(steps):
        #proposed_graph = propose_modification(current_graph.copy(), current_stats, target_stats)
        proposed_graph = propose_modification(current_graph.copy())
        proposed_cost, proposed_stats = cost(proposed_graph, target_stats, stat_train, return_stats=True)
        delta = proposed_cost - current_cost

        # Accept the move if the cost is reduced.
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_graph = proposed_graph
            current_cost = proposed_cost
            current_stats = proposed_stats
            if current_cost < best_cost:
                best_graph = current_graph.copy()

        # Cool down the system.
        temp *= cooling_rate

    return best_graph


def process_single_graph(args):
    """
    Process a single graph: load it, compute its initial cost,
    refine it, compute the refined cost, and return the result.
    
    Parameters:
      args: a tuple (graph_index, edgelist, stats, stat_train)
    Returns:
      (graph_index, refined_edgelist)
    """
    graph_index, edgelist, stats, stat_train = args

    G = nx.Graph()
    G.add_edges_from(edgelist[1])
    node_diff = stats[graph_index][0] - G.number_of_nodes()
    if node_diff > 0:
        G.add_nodes_from(range(int(stats[graph_index][0])))
    elif node_diff < 0:
        G.remove_nodes_from(range(int(stats[graph_index][0]), G.number_of_nodes()))
        print("removed nodes")
    
    mae_initial = cost(G, stats[graph_index], stat_train)
    #print(f"Initial Graph {graph_index} MAE: {mae_initial}")
    
    refined_G = refine_graph(G.copy(), stats[graph_index], stat_train)
    refined_edgelist = list(refined_G.edges())
    
    mae_refined = cost(refined_G.copy(), stats[graph_index], stat_train)
    #print(f"Refined Graph {graph_index} MAE: {mae_refined}")

    if mae_refined < mae_initial:
        return graph_index, refined_edgelist
    return graph_index, edgelist[1]
    


def refine_graphs(output_filepath="output3.csv"):
    # Load necessary training and testing statistics and data.
    stat_train = save_train_statistics()
    file_path = "output2.csv"
    edges_lists = load_edgenodes_from_csv(file_path)
    stats = save_test_statistics()

    # Prepare arguments for each graph.
    args_list = [(i, edgelist, stats, stat_train)
                 for i, edgelist in enumerate(edges_lists)]
    
    # This list will eventually hold the refined edgelists.
    refined_edgelists = [None] * len(edges_lists)

    # Use a ProcessPoolExecutor to parallelize the processing of graphs.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_single_graph, args_list),
            total=len(args_list),
            desc="Processing graphs"
        ))
    
    # Place results in the refined_edgelists list using the returned graph index.
    for graph_index, edgelist in results:
        refined_edgelists[graph_index] = edgelist

    # Save refined edgelists to a CSV file.
    with open(output_filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["graph_id", "edge_list"])
        
        for graph_id, edgelist in enumerate(tqdm(refined_edgelists, desc="Saving edgelists")):
            # Convert the edge list to a single string.
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in edgelist])
            writer.writerow([f"graph_{graph_id}", edge_list_text])

    get_score(output_filepath)


if __name__ == "__main__":
    refine_graphs()