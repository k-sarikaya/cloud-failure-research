"""
Network Topology Generators

Implements various network topology generation algorithms:
- Barabási-Albert (Scale-free)
- Watts-Strogatz (Small-world)
- Erdős-Rényi (Random)
- 5G MEC (Hierarchical)

Anonymous Submission for Peer Review
"""

import numpy as np
from typing import Tuple


def generate_barabasi_albert(n_nodes: int, m: int = 3) -> np.ndarray:
    """
    Generate Barabási-Albert preferential attachment graph.
    
    Args:
        n_nodes: Number of nodes
        m: Number of edges to attach from new node to existing nodes
    
    Returns:
        Adjacency matrix (n_nodes x n_nodes)
    """
    adj = np.zeros((n_nodes, n_nodes))
    
    # Initialize fully connected core of m+1 nodes
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = 1
            adj[j, i] = 1
    
    # Track degrees for preferential attachment
    degrees = adj.sum(axis=1)
    
    # Add remaining nodes
    for new_node in range(m + 1, n_nodes):
        # Probability proportional to degree
        probs = degrees[:new_node] / degrees[:new_node].sum()
        
        # Select m targets without replacement
        targets = np.random.choice(
            new_node, 
            size=min(m, new_node), 
            replace=False, 
            p=probs
        )
        
        # Add edges
        for t in targets:
            adj[new_node, t] = 1
            adj[t, new_node] = 1
        
        # Update degrees
        degrees = adj.sum(axis=1)
    
    return adj


def generate_watts_strogatz(n_nodes: int, k: int = 6, p: float = 0.3) -> np.ndarray:
    """
    Generate Watts-Strogatz small-world graph.
    
    Args:
        n_nodes: Number of nodes
        k: Each node connected to k nearest neighbors in ring
        p: Probability of rewiring each edge
    
    Returns:
        Adjacency matrix (n_nodes x n_nodes)
    """
    adj = np.zeros((n_nodes, n_nodes))
    
    # Create ring lattice with k neighbors
    for i in range(n_nodes):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % n_nodes
            adj[i, neighbor] = 1
            adj[neighbor, i] = 1
    
    # Rewire edges with probability p
    for i in range(n_nodes):
        for j in range(1, k // 2 + 1):
            if np.random.random() < p:
                neighbor = (i + j) % n_nodes
                
                # Remove original edge
                adj[i, neighbor] = 0
                adj[neighbor, i] = 0
                
                # Find new target (not self, not already connected)
                candidates = [
                    x for x in range(n_nodes) 
                    if x != i and adj[i, x] == 0
                ]
                
                if candidates:
                    new_neighbor = np.random.choice(candidates)
                    adj[i, new_neighbor] = 1
                    adj[new_neighbor, i] = 1
    
    return adj


def generate_erdos_renyi(n_nodes: int, p: float = 0.06) -> np.ndarray:
    """
    Generate Erdős-Rényi random graph.
    
    Args:
        n_nodes: Number of nodes
        p: Probability of edge between any two nodes
    
    Returns:
        Adjacency matrix (n_nodes x n_nodes)
    """
    adj = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < p:
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj


def generate_5g_mec(n_nodes: int = 150) -> np.ndarray:
    """
    Generate hierarchical 5G MEC topology.
    
    Structure:
    - Core layer: Fully connected data centers
    - Edge layer: Connected to 2-3 core nodes
    - Access layer: Connected to 1-2 edge nodes
    
    Args:
        n_nodes: Total number of nodes
    
    Returns:
        Adjacency matrix (n_nodes x n_nodes)
    """
    adj = np.zeros((n_nodes, n_nodes))
    
    # Layer sizes (5% core, 15% edge, 80% access)
    n_core = max(3, int(n_nodes * 0.05))
    n_edge = max(5, int(n_nodes * 0.15))
    n_access = n_nodes - n_core - n_edge
    
    # Core nodes: fully connected mesh
    for i in range(n_core):
        for j in range(i + 1, n_core):
            adj[i, j] = 1
            adj[j, i] = 1
    
    # Edge nodes: each connected to 2-3 core nodes
    edge_start = n_core
    for i in range(edge_start, edge_start + n_edge):
        n_connections = np.random.randint(2, 4)
        targets = np.random.choice(n_core, size=min(n_connections, n_core), replace=False)
        for t in targets:
            adj[i, t] = 1
            adj[t, i] = 1
    
    # Access nodes: each connected to 1-2 edge nodes
    access_start = n_core + n_edge
    for i in range(access_start, n_nodes):
        n_connections = np.random.randint(1, 3)
        edge_nodes = range(edge_start, edge_start + n_edge)
        targets = np.random.choice(
            list(edge_nodes), 
            size=min(n_connections, n_edge), 
            replace=False
        )
        for t in targets:
            adj[i, t] = 1
            adj[t, i] = 1
    
    return adj


def compute_graph_metrics(adj: np.ndarray) -> dict:
    """
    Compute basic graph metrics.
    
    Returns:
        Dictionary with: n_nodes, n_edges, avg_degree, density, clustering
    """
    n_nodes = len(adj)
    n_edges = int(adj.sum() / 2)
    avg_degree = adj.sum() / n_nodes
    density = 2 * n_edges / (n_nodes * (n_nodes - 1))
    
    # Clustering coefficient
    clustering_coeffs = []
    for i in range(n_nodes):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering_coeffs.append(0)
            continue
        
        # Count edges between neighbors
        edges = 0
        for n1 in neighbors:
            for n2 in neighbors:
                if n1 < n2 and adj[n1, n2] > 0:
                    edges += 1
        
        max_edges = k * (k - 1) / 2
        clustering_coeffs.append(edges / max_edges if max_edges > 0 else 0)
    
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'avg_degree': avg_degree,
        'density': density,
        'clustering_coefficient': np.mean(clustering_coeffs)
    }


if __name__ == "__main__":
    # Demo: Generate and analyze each topology
    np.random.seed(42)
    
    topologies = {
        'Barabási-Albert': generate_barabasi_albert(100, m=3),
        'Watts-Strogatz': generate_watts_strogatz(100, k=6, p=0.3),
        'Erdős-Rényi': generate_erdos_renyi(100, p=0.06),
        '5G MEC': generate_5g_mec(150)
    }
    
    print("Topology Analysis")
    print("=" * 60)
    
    for name, adj in topologies.items():
        metrics = compute_graph_metrics(adj)
        print(f"\n{name}:")
        print(f"  Nodes: {metrics['n_nodes']}")
        print(f"  Edges: {metrics['n_edges']}")
        print(f"  Avg Degree: {metrics['avg_degree']:.2f}")
        print(f"  Density: {metrics['density']:.4f}")
        print(f"  Clustering: {metrics['clustering_coefficient']:.4f}")
