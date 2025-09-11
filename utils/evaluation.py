"""
Evaluation metrics for Visual Place Recognition
Implements standard VPR metrics including Recall@K and others
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import average_precision_score
import faiss
from tqdm import tqdm


def compute_distance_matrix(
    query_features: np.ndarray,
    database_features: np.ndarray,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute distance matrix between query and database features.
    
    Args:
        query_features: Query feature vectors (N_q, D)
        database_features: Database feature vectors (N_db, D)
        metric: Distance metric ('cosine', 'euclidean', 'dot')
    
    Returns:
        Distance matrix (N_q, N_db)
    """
    if metric == 'cosine':
        # Normalize features
        query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
        database_features = database_features / np.linalg.norm(database_features, axis=1, keepdims=True)
        
        # Compute cosine similarity and convert to distance
        similarity = np.dot(query_features, database_features.T)
        distances = 1 - similarity
        
    elif metric == 'euclidean':
        # Use FAISS for efficient computation
        index = faiss.IndexFlatL2(database_features.shape[1])
        index.add(database_features.astype(np.float32))
        distances, _ = index.search(query_features.astype(np.float32), database_features.shape[0])
        
    elif metric == 'dot':
        # Dot product (higher is better, so negate)
        distances = -np.dot(query_features, database_features.T)
        
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def get_ground_truth_matches(
    query_gps: np.ndarray,
    database_gps: np.ndarray,
    threshold: float = 25.0
) -> List[List[int]]:
    """
    Get ground truth matches based on GPS coordinates.
    
    Args:
        query_gps: Query GPS coordinates (N_q, 2) [lat, lon]
        database_gps: Database GPS coordinates (N_db, 2) [lat, lon]
        threshold: Distance threshold in meters for positive match
    
    Returns:
        List of lists containing ground truth matches for each query
    """
    from datasets.gps_dataset import haversine_distance
    
    ground_truth = []
    
    for i, (q_lat, q_lon) in enumerate(query_gps):
        matches = []
        for j, (db_lat, db_lon) in enumerate(database_gps):
            distance = haversine_distance(q_lat, q_lon, db_lat, db_lon)
            if distance <= threshold:
                matches.append(j)
        ground_truth.append(matches)
    
    return ground_truth


def recall_at_k(
    distances: np.ndarray,
    ground_truth: List[List[int]],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[int, float]:
    """
    Compute Recall@K for different values of K.
    
    Args:
        distances: Distance matrix (N_q, N_db)
        ground_truth: Ground truth matches for each query
        k_values: List of K values to compute recall for
    
    Returns:
        Dictionary mapping K to Recall@K
    """
    n_queries = distances.shape[0]
    recalls = {}
    
    # Get ranked indices (sorted by distance)
    ranked_indices = np.argsort(distances, axis=1)
    
    for k in k_values:
        correct = 0
        for i in range(n_queries):
            if len(ground_truth[i]) == 0:
                continue
            
            # Get top-k predictions
            top_k = ranked_indices[i, :k]
            
            # Check if any prediction is correct
            if any(pred in ground_truth[i] for pred in top_k):
                correct += 1
        
        recalls[k] = correct / n_queries if n_queries > 0 else 0.0
    
    return recalls


def precision_at_k(
    distances: np.ndarray,
    ground_truth: List[List[int]],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[int, float]:
    """
    Compute Precision@K for different values of K.
    """
    n_queries = distances.shape[0]
    precisions = {}
    
    ranked_indices = np.argsort(distances, axis=1)
    
    for k in k_values:
        total_precision = 0
        valid_queries = 0
        
        for i in range(n_queries):
            if len(ground_truth[i]) == 0:
                continue
            
            top_k = ranked_indices[i, :k]
            correct_in_top_k = sum(1 for pred in top_k if pred in ground_truth[i])
            
            total_precision += correct_in_top_k / k
            valid_queries += 1
        
        precisions[k] = total_precision / valid_queries if valid_queries > 0 else 0.0
    
    return precisions


def average_precision(
    distances: np.ndarray,
    ground_truth: List[List[int]]
) -> float:
    """
    Compute mean Average Precision (mAP).
    """
    n_queries = distances.shape[0]
    n_database = distances.shape[1]
    
    aps = []
    
    for i in range(n_queries):
        if len(ground_truth[i]) == 0:
            continue
        
        # Create binary relevance vector
        relevance = np.zeros(n_database)
        relevance[ground_truth[i]] = 1
        
        # Get ranking based on distances (lower is better)
        ranking = np.argsort(distances[i])
        ranked_relevance = relevance[ranking]
        
        # Compute average precision
        ap = average_precision_score(ranked_relevance, -distances[i][ranking])
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


def evaluate_vpr(
    query_features: np.ndarray,
    database_features: np.ndarray,
    query_gps: np.ndarray,
    database_gps: np.ndarray,
    distance_threshold: float = 25.0,
    k_values: List[int] = [1, 5, 10, 20],
    metric: str = 'cosine'
) -> Dict:
    """
    Complete VPR evaluation pipeline.
    
    Args:
        query_features: Query feature vectors (N_q, D)
        database_features: Database feature vectors (N_db, D)
        query_gps: Query GPS coordinates (N_q, 2)
        database_gps: Database GPS coordinates (N_db, 2)
        distance_threshold: GPS distance threshold for positive matches
        k_values: K values for Recall@K computation
        metric: Distance metric to use
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    
    print("Computing distance matrix...")
    distances = compute_distance_matrix(query_features, database_features, metric)
    
    print("Getting ground truth matches...")
    ground_truth = get_ground_truth_matches(query_gps, database_gps, distance_threshold)
    
    print("Computing metrics...")
    recalls = recall_at_k(distances, ground_truth, k_values)
    precisions = precision_at_k(distances, ground_truth, k_values)
    map_score = average_precision(distances, ground_truth)
    
    # Count statistics
    total_queries = len(query_features)
    queries_with_matches = sum(1 for gt in ground_truth if len(gt) > 0)
    avg_matches_per_query = np.mean([len(gt) for gt in ground_truth])
    
    results = {
        'recall_at_k': recalls,
        'precision_at_k': precisions,
        'mean_average_precision': map_score,
        'total_queries': total_queries,
        'queries_with_matches': queries_with_matches,
        'avg_matches_per_query': avg_matches_per_query,
        'distance_threshold': distance_threshold,
        'metric': metric
    }
    
    return results


def print_evaluation_results(results: Dict):
    """
    Print evaluation results in a nice format.
    """
    print("\n" + "="*60)
    print("VPR EVALUATION RESULTS")
    print("="*60)
    
    print(f"Total queries: {results['total_queries']}")
    print(f"Queries with matches: {results['queries_with_matches']}")
    print(f"Average matches per query: {results['avg_matches_per_query']:.2f}")
    print(f"Distance threshold: {results['distance_threshold']}m")
    print(f"Distance metric: {results['metric']}")
    
    print("\nRecall@K:")
    for k, recall in results['recall_at_k'].items():
        print(f"  Recall@{k}: {recall:.3f}")
    
    print("\nPrecision@K:")
    for k, precision in results['precision_at_k'].items():
        print(f"  Precision@{k}: {precision:.3f}")
    
    print(f"\nmAP: {results['mean_average_precision']:.3f}")
    print("="*60)


class VPREvaluator:
    """
    Class for evaluating VPR models with different configurations.
    """
    
    def __init__(
        self,
        distance_threshold: float = 25.0,
        k_values: List[int] = [1, 5, 10, 20],
        metric: str = 'cosine'
    ):
        self.distance_threshold = distance_threshold
        self.k_values = k_values
        self.metric = metric
    
    def evaluate(
        self,
        query_features: np.ndarray,
        database_features: np.ndarray,
        query_gps: np.ndarray,
        database_gps: np.ndarray
    ) -> Dict:
        """Evaluate VPR performance."""
        return evaluate_vpr(
            query_features=query_features,
            database_features=database_features,
            query_gps=query_gps,
            database_gps=database_gps,
            distance_threshold=self.distance_threshold,
            k_values=self.k_values,
            metric=self.metric
        )
    
    def evaluate(
        self,
        all_features: np.ndarray,
        all_locations: List[Tuple[float, float]],
        distance_metric: str = None
    ) -> Dict:
        """
        Evaluate VPR performance using all features as both query and database.
        
        Args:
            all_features: All feature vectors (N, D)
            all_locations: All GPS locations [(lat, lon), ...]
            distance_metric: Distance metric override (optional)
        
        Returns:
            Evaluation results dictionary
        """
        # Use override metric if provided, otherwise use default
        metric = distance_metric if distance_metric is not None else self.metric
        
        # Convert locations to numpy array
        all_gps = np.array(all_locations)
        
        # Use all features as both query and database
        return evaluate_vpr(
            query_features=all_features,
            database_features=all_features,
            query_gps=all_gps,
            database_gps=all_gps,
            distance_threshold=self.distance_threshold,
            k_values=self.k_values,
            metric=metric
        )
    
    def compare_models(
        self,
        model_results: Dict[str, Dict]
    ) -> Dict:
        """
        Compare multiple models and return a summary table.
        
        Args:
            model_results: Dictionary mapping model names to their evaluation results
        
        Returns:
            Comparison summary
        """
        comparison = {
            'models': list(model_results.keys()),
            'recall_at_1': [],
            'recall_at_5': [],
            'recall_at_10': [],
            'recall_at_20': [],
            'map': []
        }
        
        for model_name, results in model_results.items():
            comparison['recall_at_1'].append(results['recall_at_k'].get(1, 0.0))
            comparison['recall_at_5'].append(results['recall_at_k'].get(5, 0.0))
            comparison['recall_at_10'].append(results['recall_at_k'].get(10, 0.0))
            comparison['recall_at_20'].append(results['recall_at_k'].get(20, 0.0))
            comparison['map'].append(results['mean_average_precision'])
        
        return comparison
    
    def print_comparison(self, comparison: Dict):
        """Print model comparison table."""
        from prettytable import PrettyTable
        
        table = PrettyTable()
        table.field_names = ["Model", "R@1", "R@5", "R@10", "R@20", "mAP"]
        
        for i, model in enumerate(comparison['models']):
            table.add_row([
                model,
                f"{comparison['recall_at_1'][i]:.3f}",
                f"{comparison['recall_at_5'][i]:.3f}",
                f"{comparison['recall_at_10'][i]:.3f}",
                f"{comparison['recall_at_20'][i]:.3f}",
                f"{comparison['map'][i]:.3f}"
            ])
        
        print("\nMODEL COMPARISON")
        print(table)
