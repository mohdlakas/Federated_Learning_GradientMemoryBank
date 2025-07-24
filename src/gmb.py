import torch
import numpy as np
from typing import Dict, List, Tuple
import faiss
import pickle
from collections import defaultdict

class GradientMemoryBank:
    """
    Gradient Memory Bank with Vector Database for Federated Learning
    
    Features:
    - Stores gradient embeddings with quality metrics
    - Uses FAISS for efficient similarity search
    - Tracks client reliability over time
    - Provides intelligent gradient weighting
    """
    
    def __init__(self, embedding_dim: int = 512, max_memories: int = 1000, 
                 similarity_threshold: float = 0.7, device: str = 'cpu'):
        """
        Initialize Gradient Memory Bank
        
        Args:
            embedding_dim: Dimension of gradient embeddings
            max_memories: Maximum number of memories to store
            similarity_threshold: Threshold for similar gradient matching
            device: Device for computations
        """
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # FAISS vector database for fast similarity search
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product similarity
        
        # Memory storage
        self.gradient_embeddings = []  # List of embeddings
        self.quality_metrics = []      # Quality scores for each gradient
        self.client_ids = []           # Which client contributed each gradient
        self.round_numbers = []        # When each gradient was added
        self.data_sizes = []           # Local data size for each client
        
        # Client reliability tracking
        self.client_reliability = defaultdict(list)  # client_id -> [quality_scores]
        self.client_participation = defaultdict(int)  # client_id -> participation_count
        
        print(f"Gradient Memory Bank initialized with {embedding_dim}D embeddings")
    
    def compute_gradient_embedding(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute embedding representation of gradients with NaN handling
        
        Args:
            gradients: Dictionary of layer_name -> gradient_tensor
            
        Returns:
            embedding: Fixed-size embedding vector
        """
        try:
            # Filter out invalid gradients
            valid_gradients = {}
            for layer_name, grad in gradients.items():
                if grad is not None and not torch.isnan(grad).any() and not torch.isinf(grad).any():
                    valid_gradients[layer_name] = grad
            
            if not valid_gradients:
                # Return zero embedding if no valid gradients
                return torch.zeros(self.embedding_dim, dtype=torch.float32)
            
            # Flatten all valid gradients and compute statistical features
            flat_grads = []
            for layer_name, grad in valid_gradients.items():
                flat_grads.append(grad.flatten())
            
            if not flat_grads:
                return torch.zeros(self.embedding_dim, dtype=torch.float32)
            
            all_grads = torch.cat(flat_grads)
            
            # Compute statistical features with safety checks
            features = []
            
            # Basic statistics with NaN checks
            mean_val = torch.mean(all_grads).item()
            std_val = torch.std(all_grads).item()
            min_val = torch.min(all_grads).item()
            max_val = torch.max(all_grads).item()
            l2_norm = torch.norm(all_grads, p=2).item()
            l1_norm = torch.norm(all_grads, p=1).item()
            
            # Validate all statistics
            stats = [mean_val, std_val, min_val, max_val, l2_norm, l1_norm]
            for stat in stats:
                if np.isnan(stat) or np.isinf(stat):
                    features.append(0.0)
                else:
                    features.append(stat)
            
            # Percentiles with safety
            try:
                sorted_grads, _ = torch.sort(all_grads)
                n = len(sorted_grads)
                percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
                for p in percentiles:
                    idx = min(int(p * n), n - 1)
                    val = sorted_grads[idx].item()
                    if np.isnan(val) or np.isinf(val):
                        features.append(0.0)
                    else:
                        features.append(val)
            except Exception:
                # Fallback percentiles
                features.extend([0.0] * 5)
            
            # Layer-wise statistics with safety
            for layer_name, grad in valid_gradients.items():
                try:
                    layer_mean = torch.mean(grad).item()
                    layer_std = torch.std(grad).item()
                    layer_norm = torch.norm(grad).item()
                    
                    # Validate layer statistics
                    layer_stats = [layer_mean, layer_std, layer_norm]
                    for stat in layer_stats:
                        if np.isnan(stat) or np.isinf(stat):
                            features.append(0.0)
                        else:
                            features.append(stat)
                except Exception:
                    features.extend([0.0, 0.0, 0.0])
                    
                # Limit features to prevent excessive memory usage
                if len(features) >= self.embedding_dim - 10:
                    break
            
            # Pad or truncate to fixed size
            features = features[:self.embedding_dim]
            if len(features) < self.embedding_dim:
                features.extend([0.0] * (self.embedding_dim - len(features)))
            
            embedding = torch.tensor(features, dtype=torch.float32)
            
            # Final NaN check
            if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                return torch.zeros(self.embedding_dim, dtype=torch.float32)
            
            return embedding
            
        except Exception as e:
            print(f"⚠️  Error computing gradient embedding: {e}")
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
    
    def compute_gradient_quality(self, gradients: Dict[str, torch.Tensor], 
                                loss_improvement: float, data_size: int) -> float:
        """
        Compute quality metric for gradients - SAFE VERSION
        
        Args:
            gradients: Client gradients
            loss_improvement: How much loss improved with these gradients
            data_size: Size of client's local dataset
            
        Returns:
            quality_score: Quality metric (higher = better)
        """
        try:
            # CRITICAL: Input validation and bounds
            if np.isnan(loss_improvement) or np.isinf(loss_improvement):
                loss_improvement = 0.0
            
            # Bound loss_improvement to reasonable range
            loss_improvement = max(0.0, min(loss_improvement, 1.0))
            
            # Compute gradient norm safely
            total_norm_sq = 0.0
            valid_params = 0
            
            for name, grad in gradients.items():
                if grad is None:
                    continue
                    
                # Check for invalid gradients
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    continue
                
                param_norm = torch.norm(grad).item()
                
                # Validate the norm
                if np.isnan(param_norm) or np.isinf(param_norm):
                    continue
                    
                total_norm_sq += param_norm ** 2
                valid_params += 1
            
            if valid_params == 0:
                return 0.1  # Default low quality
            
            total_norm = np.sqrt(total_norm_sq)
            
            # FIXED: Simple, bounded quality computation
            # 1. Gradient norm quality (prefer moderate norms around 1.0)
            if total_norm > 0:
                # Use a simple function that prefers norms around 1.0
                norm_quality = 1.0 / (1.0 + abs(total_norm - 1.0))
                norm_quality = max(0.0, min(1.0, norm_quality))
            else:
                norm_quality = 0.1
            
            # 2. Loss improvement quality (already bounded)
            loss_quality = loss_improvement
            
            # 3. Data size quality
            data_quality = min(1.0, max(0.0, data_size / 1000.0))
            
            # Simple weighted average - NO COMPLEX MATH
            quality_score = 0.4 * norm_quality + 0.4 * loss_quality + 0.2 * data_quality
            
            # CRITICAL: Final bounds check
            quality_score = max(0.0, min(1.0, quality_score))
            
            # CRITICAL: NaN/inf check
            if np.isnan(quality_score) or np.isinf(quality_score):
                quality_score = 0.1
            
            return quality_score
            
        except Exception as e:
            print(f"⚠️  Exception in quality computation: {e}")
            return 0.1  # Safe fallback
    
    def add_gradient_memory(self, client_id: int, gradients: Dict[str, torch.Tensor],
                           loss_improvement: float, data_size: int, round_num: int):
        """
        Add new gradient memory to the bank with safety checks
        
        Args:
            client_id: ID of contributing client
            gradients: Client gradients
            loss_improvement: Loss improvement from these gradients
            data_size: Client's local data size
            round_num: Current communication round
        """
        try:
            # Compute embedding and quality with safety checks
            embedding = self.compute_gradient_embedding(gradients)
            quality = self.compute_gradient_quality(gradients, loss_improvement, data_size)
            
            # Validate embedding and quality
            if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                print(f"⚠️  Invalid embedding for client {client_id}, skipping")
                return
            
            if np.isnan(quality) or np.isinf(quality):
                print(f"⚠️  Invalid quality for client {client_id}, using default")
                quality = 0.1
            
            # Store memory
            self.gradient_embeddings.append(embedding.numpy())
            self.quality_metrics.append(quality)
            self.client_ids.append(client_id)
            self.round_numbers.append(round_num)
            self.data_sizes.append(data_size)
            
            # Update client reliability
            self.client_reliability[client_id].append(quality)
            self.client_participation[client_id] += 1
            
            # Add to FAISS index with safety
            try:
                self.index.add(embedding.numpy().reshape(1, -1))
            except Exception as e:
                print(f"⚠️  Error adding to FAISS index: {e}")
            
            # Maintain memory limit
            if len(self.gradient_embeddings) > self.max_memories:
                self._remove_oldest_memory()
            
            print(f"Added gradient memory from client {client_id} with quality {quality:.3f}")
            
        except Exception as e:
            print(f"⚠️  Error adding gradient memory for client {client_id}: {e}")
    
    def get_similar_gradients(self, query_embedding: torch.Tensor, 
                             top_k: int = 10) -> List[Tuple[int, float, float]]:
        """
        Find similar historical gradients with safety checks
        
        Args:
            query_embedding: Embedding to search for
            top_k: Number of similar gradients to return
            
        Returns:
            List of (memory_index, similarity_score, quality_score)
        """
        try:
            if self.index.ntotal == 0:
                return []
            
            # Validate query embedding
            if torch.isnan(query_embedding).any() or torch.isinf(query_embedding).any():
                return []
            
            # Search for similar embeddings
            query = query_embedding.numpy().reshape(1, -1)
            similarities, indices = self.index.search(query, min(top_k, self.index.ntotal))
            
            results = []
            for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                if not (np.isnan(sim) or np.isinf(sim)) and sim > self.similarity_threshold:
                    quality = self.quality_metrics[idx]
                    if not (np.isnan(quality) or np.isinf(quality)):
                        results.append((idx, sim, quality))
            
            return results
            
        except Exception as e:
            print(f"⚠️  Error finding similar gradients: {e}")
            return []
    
    def compute_client_weights(self, participating_clients: List[int], 
                              current_gradients: Dict[int, Dict[str, torch.Tensor]],
                              current_data_sizes: Dict[int, int]) -> Dict[int, float]:
        """
        Compute intelligent weights for client aggregation - FIXED VERSION
        
        Args:
            participating_clients: List of client IDs in current round
            current_gradients: Dict of client_id -> gradients
            current_data_sizes: Dict of client_id -> data_size
            
        Returns:
            Dict of client_id -> aggregation_weight
        """
        try:
            weights = {}
            
            # Get max data size for normalization
            max_data_size = max(current_data_sizes.values()) if current_data_sizes else 1
            if max_data_size <= 0:
                max_data_size = 1
            
            for client_id in participating_clients:
                # Base weight from data size (normalized)
                data_weight = current_data_sizes.get(client_id, 1) / max_data_size
                
                # Historical reliability weight
                if client_id in self.client_reliability and len(self.client_reliability[client_id]) > 0:
                    # FIXED: Handle potential inf/nan in reliability scores
                    client_qualities = self.client_reliability[client_id]
                    valid_qualities = [q for q in client_qualities if not (np.isnan(q) or np.isinf(q))]
                    
                    if valid_qualities:
                        avg_quality = np.mean(valid_qualities)
                        # Bound the quality
                        avg_quality = max(0.0, min(1.0, avg_quality))
                        reliability_weight = avg_quality
                    else:
                        reliability_weight = 0.5  # Default for invalid qualities
                else:
                    reliability_weight = 0.5  # Default for new clients
                
                # Similarity weight (simplified to avoid complex computation)
                similarity_weight = 0.5  # Default - remove complex similarity computation for now
                
                # Combined weight with bounds checking
                final_weight = 0.4 * data_weight + 0.3 * reliability_weight + 0.3 * similarity_weight
                
                # CRITICAL: Ensure weight is valid
                if np.isnan(final_weight) or np.isinf(final_weight) or final_weight <= 0:
                    final_weight = 1.0  # Default weight
                
                weights[client_id] = final_weight
            
            # FIXED: Safe weight normalization
            total_weight = sum(weights.values())
            
            if total_weight <= 0 or np.isnan(total_weight) or np.isinf(total_weight):
                # Fallback to uniform weights
                uniform_weight = 1.0 / len(participating_clients)
                weights = {k: uniform_weight for k in participating_clients}
            else:
                # Normalize weights safely
                normalized_weights = {}
                for k, v in weights.items():
                    norm_weight = v / total_weight
                    if np.isnan(norm_weight) or np.isinf(norm_weight) or norm_weight <= 0:
                        norm_weight = 1.0 / len(participating_clients)
                    normalized_weights[k] = norm_weight
                weights = normalized_weights
            
            return weights
            
        except Exception as e:
            print(f"⚠️  Error computing client weights: {e}")
            # Fallback to uniform weights
            uniform_weight = 1.0 / len(participating_clients)
            return {k: uniform_weight for k in participating_clients}
    
    def get_client_reliability_stats(self) -> Dict[int, Dict[str, float]]:
        """Get reliability statistics for all clients with safety checks"""
        stats = {}
        for client_id, qualities in self.client_reliability.items():
            try:
                valid_qualities = [q for q in qualities if not (np.isnan(q) or np.isinf(q))]
                
                if valid_qualities:
                    avg_quality = np.mean(valid_qualities)
                    std_quality = np.std(valid_qualities)
                    participation = self.client_participation[client_id]
                    
                    # Safe reliability score computation
                    reliability_score = avg_quality * np.log(1 + participation)
                    
                    # Validate all stats
                    if np.isnan(avg_quality) or np.isinf(avg_quality):
                        avg_quality = 0.5
                    if np.isnan(std_quality) or np.isinf(std_quality):
                        std_quality = 0.0
                    if np.isnan(reliability_score) or np.isinf(reliability_score):
                        reliability_score = 0.5
                    
                    stats[client_id] = {
                        'avg_quality': avg_quality,
                        'std_quality': std_quality,
                        'participation_count': participation,
                        'reliability_score': reliability_score
                    }
                else:
                    # Default stats for clients with no valid qualities
                    stats[client_id] = {
                        'avg_quality': 0.5,
                        'std_quality': 0.0,
                        'participation_count': self.client_participation[client_id],
                        'reliability_score': 0.5
                    }
            except Exception as e:
                print(f"⚠️  Error computing stats for client {client_id}: {e}")
                stats[client_id] = {
                    'avg_quality': 0.5,
                    'std_quality': 0.0,
                    'participation_count': 1,
                    'reliability_score': 0.5
                }
        
        return stats
    
    def _remove_oldest_memory(self):
        """Remove oldest memory to maintain size limit"""
        try:
            if len(self.gradient_embeddings) > 0:
                # Remove from all lists
                self.gradient_embeddings.pop(0)
                self.quality_metrics.pop(0)
                removed_client = self.client_ids.pop(0)
                self.round_numbers.pop(0)
                self.data_sizes.pop(0)
                
                # Rebuild FAISS index (expensive but necessary)
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                if self.gradient_embeddings:
                    embeddings_array = np.array(self.gradient_embeddings)
                    self.index.add(embeddings_array)
                
                print(f"Removed oldest memory from client {removed_client}")
        except Exception as e:
            print(f"⚠️  Error removing oldest memory: {e}")
    
    def save_memory_bank(self, filepath: str):
        """Save memory bank to disk"""
        try:
            data = {
                'gradient_embeddings': self.gradient_embeddings,
                'quality_metrics': self.quality_metrics,
                'client_ids': self.client_ids,
                'round_numbers': self.round_numbers,
                'data_sizes': self.data_sizes,
                'client_reliability': dict(self.client_reliability),
                'client_participation': dict(self.client_participation),
                'embedding_dim': self.embedding_dim,
                'max_memories': self.max_memories,
                'similarity_threshold': self.similarity_threshold
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"Memory bank saved to {filepath}")
        except Exception as e:
            print(f"⚠️  Error saving memory bank: {e}")
    
    def load_memory_bank(self, filepath: str):
        """Load memory bank from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.gradient_embeddings = data['gradient_embeddings']
            self.quality_metrics = data['quality_metrics']
            self.client_ids = data['client_ids']
            self.round_numbers = data['round_numbers']
            self.data_sizes = data['data_sizes']
            self.client_reliability = defaultdict(list, data['client_reliability'])
            self.client_participation = defaultdict(int, data['client_participation'])
            
            # Rebuild FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            if self.gradient_embeddings:
                embeddings_array = np.array(self.gradient_embeddings)
                self.index.add(embeddings_array)
            
            print(f"Memory bank loaded from {filepath}")
        except Exception as e:
            print(f"⚠️  Error loading memory bank: {e}")
