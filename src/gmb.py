import torch
import numpy as np
from typing import Dict, List, Tuple
import faiss
import pickle
from collections import defaultdict

class GradientMemoryBank:
    """
    Gradient Memory Bank with Vector Database for Federated Learning - ADAM OPTIMIZED
    
    Features:
    - Stores parameter update embeddings with quality metrics for Adam optimizer
    - Uses FAISS for efficient similarity search
    - Tracks client reliability over time based on loss improvements
    - Provides intelligent update weighting for Adam-based federated learning
    """
    
    def __init__(self, embedding_dim: int = 512, max_memories: int = 1000, 
                 similarity_threshold: float = 0.7, device: str = 'cpu'):
        """
        Initialize Gradient Memory Bank for Adam Optimizer
        
        Args:
            embedding_dim: Dimension of parameter update embeddings
            max_memories: Maximum number of memories to store
            similarity_threshold: Threshold for similar update matching
            device: Device for computations
        """
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # FAISS vector database for fast similarity search
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product similarity
        
        # Memory storage
        self.update_embeddings = []        # List of parameter update embeddings
        self.quality_metrics = []          # Quality scores for each update
        self.client_ids = []               # Which client contributed each update
        self.round_numbers = []            # When each update was added
        self.data_sizes = []               # Local data size for each client
        self.loss_improvements = []        # Loss improvement for each update
        
        # Client reliability tracking
        self.client_reliability = defaultdict(list)  # client_id -> [quality_scores]
        self.client_participation = defaultdict(int)  # client_id -> participation_count
        self.client_loss_history = defaultdict(list)  # client_id -> [loss_improvements]
        
        print(f"Adam-optimized Gradient Memory Bank initialized with {embedding_dim}D embeddings")
    
    def compute_update_embedding(self, param_updates: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute embedding representation of parameter updates for Adam optimizer
        
        Args:
            param_updates: Dictionary of layer_name -> parameter_update_tensor
            
        Returns:
            embedding: Fixed-size embedding vector
        """
        try:
            # Filter out invalid updates
            valid_updates = {}
            for layer_name, update in param_updates.items():
                if update is not None and not torch.isnan(update).any() and not torch.isinf(update).any():
                    valid_updates[layer_name] = update
            
            if not valid_updates:
                return torch.zeros(self.embedding_dim, dtype=torch.float32)
            
            # Flatten all valid updates
            flat_updates = []
            for layer_name, update in valid_updates.items():
                flat_updates.append(update.flatten())
            
            if not flat_updates:
                return torch.zeros(self.embedding_dim, dtype=torch.float32)
            
            all_updates = torch.cat(flat_updates)
            
            # Compute Adam-specific statistical features
            features = []
            
            # Basic statistics
            mean_val = torch.mean(all_updates).item()
            std_val = torch.std(all_updates).item()
            min_val = torch.min(all_updates).item()
            max_val = torch.max(all_updates).item()
            l2_norm = torch.norm(all_updates, p=2).item()
            l1_norm = torch.norm(all_updates, p=1).item()
            
            # Adam-specific features
            abs_mean = torch.mean(torch.abs(all_updates)).item()
            median_val = torch.median(all_updates).item()
            
            # Update magnitude distribution
            update_magnitudes = torch.abs(all_updates)
            mag_mean = torch.mean(update_magnitudes).item()
            mag_std = torch.std(update_magnitudes).item()
            
            # Validate all statistics
            stats = [mean_val, std_val, min_val, max_val, l2_norm, l1_norm, 
                    abs_mean, median_val, mag_mean, mag_std]
            
            for stat in stats:
                if np.isnan(stat) or np.isinf(stat):
                    features.append(0.0)
                else:
                    features.append(stat)
            
            # Percentiles for update distribution
            try:
                sorted_updates, _ = torch.sort(all_updates)
                n = len(sorted_updates)
                percentiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
                for p in percentiles:
                    idx = min(int(p * n), n - 1)
                    val = sorted_updates[idx].item()
                    if np.isnan(val) or np.isinf(val):
                        features.append(0.0)
                    else:
                        features.append(val)
            except Exception:
                features.extend([0.0] * 7)
            
            # Layer-wise update characteristics
            for layer_name, update in valid_updates.items():
                try:
                    layer_mean = torch.mean(update).item()
                    layer_std = torch.std(update).item()
                    layer_norm = torch.norm(update).item()
                    layer_abs_mean = torch.mean(torch.abs(update)).item()
                    
                    layer_stats = [layer_mean, layer_std, layer_norm, layer_abs_mean]
                    for stat in layer_stats:
                        if np.isnan(stat) or np.isinf(stat):
                            features.append(0.0)
                        else:
                            features.append(stat)
                except Exception:
                    features.extend([0.0, 0.0, 0.0, 0.0])
                    
                # Limit features to prevent excessive memory usage
                if len(features) >= self.embedding_dim - 10:
                    break
            
            # Pad or truncate to fixed size
            features = features[:self.embedding_dim]
            if len(features) < self.embedding_dim:
                features.extend([0.0] * (self.embedding_dim - len(features)))
            
            embedding = torch.tensor(features, dtype=torch.float32)
            
            # Final validation
            if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                return torch.zeros(self.embedding_dim, dtype=torch.float32)
            
            return embedding
            
        except Exception as e:
            print(f"⚠️  Error computing update embedding: {e}")
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
    
    def compute_adam_quality(self, param_updates: Dict[str, torch.Tensor], 
                            loss_improvement: float, data_size: int) -> float:
        """
        Compute quality metric for Adam parameter updates
        
        Args:
            param_updates: Client parameter updates
            loss_improvement: How much loss improved with these updates
            data_size: Size of client's local dataset
            
        Returns:
            quality_score: Quality metric (higher = better)
        """
        try:
            # Input validation
            if np.isnan(loss_improvement) or np.isinf(loss_improvement):
                loss_improvement = 0.0
            
            # Bound loss improvement
            loss_improvement = max(-2.0, min(loss_improvement, 2.0))
            
            # 1. Loss improvement quality (primary factor for Adam)
            if loss_improvement > 0:
                # Positive improvement - scale based on magnitude
                loss_quality = min(1.0, loss_improvement * 5.0)  # Scale up improvements
            elif loss_improvement == 0:
                loss_quality = 0.2  # Neutral but low
            else:
                # Negative improvement (loss got worse) - penalize but don't eliminate
                loss_quality = max(0.0, 0.2 + loss_improvement * 0.5)
            
            # 2. Update consistency quality
            total_update_norm = 0.0
            valid_params = 0
            update_consistency = 0.0
            
            for name, update in param_updates.items():
                if update is None or torch.isnan(update).any() or torch.isinf(update).any():
                    continue
                
                param_norm = torch.norm(update).item()
                if np.isnan(param_norm) or np.isinf(param_norm):
                    continue
                    
                total_update_norm += param_norm ** 2
                
                # Measure update consistency (prefer updates that are not too sparse)
                non_zero_ratio = (torch.abs(update) > 1e-8).float().mean().item()
                update_consistency += non_zero_ratio
                
                valid_params += 1
            
            if valid_params == 0:
                update_quality = 0.1
            else:
                total_update_norm = np.sqrt(total_update_norm)
                update_consistency = update_consistency / valid_params
                
                # For Adam, prefer moderate update norms
                if total_update_norm > 0:
                    if 0.0001 <= total_update_norm <= 0.1:
                        norm_quality = 1.0
                    elif total_update_norm < 0.0001:
                        norm_quality = total_update_norm / 0.0001
                    else:
                        norm_quality = 0.1 / total_update_norm
                else:
                    norm_quality = 0.1
                
                # Combine norm and consistency
                update_quality = 0.7 * norm_quality + 0.3 * update_consistency
            
            # 3. Data size quality (minor factor)
            data_quality = min(1.0, max(0.1, np.log(data_size + 1) / np.log(1000)))
            
            # Weighted combination - prioritize loss improvement for Adam
            quality_score = 0.8 * loss_quality + 0.15 * update_quality + 0.05 * data_quality
            
            # Final bounds and validation
            quality_score = max(0.0, min(1.0, quality_score))
            
            if np.isnan(quality_score) or np.isinf(quality_score):
                quality_score = 0.1
            
            return quality_score
            
        except Exception as e:
            print(f"⚠️  Exception in Adam quality computation: {e}")
            return 0.1
    
    def add_update_memory(self, client_id: int, param_updates: Dict[str, torch.Tensor],
                         loss_improvement: float, data_size: int, round_num: int):
        """
        Add new parameter update memory to the bank
        
        Args:
            client_id: ID of contributing client
            param_updates: Client parameter updates from Adam optimizer
            loss_improvement: Loss improvement from these updates
            data_size: Client's local data size
            round_num: Current communication round
        """
        try:
            # Compute embedding and quality
            embedding = self.compute_update_embedding(param_updates)
            quality = self.compute_adam_quality(param_updates, loss_improvement, data_size)
            
            # Validate embedding and quality
            if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                print(f"⚠️  Invalid embedding for client {client_id}, skipping")
                return
            
            if np.isnan(quality) or np.isinf(quality):
                print(f"⚠️  Invalid quality for client {client_id}, using default")
                quality = 0.1
            
            # Store memory
            self.update_embeddings.append(embedding.numpy())
            self.quality_metrics.append(quality)
            self.client_ids.append(client_id)
            self.round_numbers.append(round_num)
            self.data_sizes.append(data_size)
            self.loss_improvements.append(loss_improvement)
            
            # Update client tracking
            self.client_reliability[client_id].append(quality)
            self.client_participation[client_id] += 1
            self.client_loss_history[client_id].append(loss_improvement)
            
            # Add to FAISS index
            try:
                self.index.add(embedding.numpy().reshape(1, -1))
            except Exception as e:
                print(f"⚠️  Error adding to FAISS index: {e}")
            
            # Maintain memory limit
            if len(self.update_embeddings) > self.max_memories:
                self._remove_oldest_memory()
            
            print(f"Added update memory from client {client_id} with quality {quality:.3f} (loss_imp: {loss_improvement:.4f})")
            
        except Exception as e:
            print(f"⚠️  Error adding update memory for client {client_id}: {e}")
    
    def get_similar_updates(self, query_embedding: torch.Tensor, 
                           top_k: int = 10) -> List[Tuple[int, float, float]]:
        """
        Find similar historical parameter updates
        
        Args:
            query_embedding: Embedding to search for
            top_k: Number of similar updates to return
            
        Returns:
            List of (memory_index, similarity_score, quality_score)
        """
        try:
            if self.index.ntotal == 0:
                return []
            
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
            print(f"⚠️  Error finding similar updates: {e}")
            return []
    
    def compute_client_weights(self, participating_clients: List[int], 
                              current_updates: Dict[int, Dict[str, torch.Tensor]],
                              current_data_sizes: Dict[int, int],
                              current_loss_improvements: Dict[int, float]) -> Dict[int, float]:
        """
        Compute intelligent weights for client aggregation - ADAM OPTIMIZED
        
        Args:
            participating_clients: List of client IDs in current round
            current_updates: Dict of client_id -> parameter_updates
            current_data_sizes: Dict of client_id -> data_size
            current_loss_improvements: Dict of client_id -> loss_improvement
            
        Returns:
            Dict of client_id -> aggregation_weight
        """
        try:
            weights = {}
            
            # Get normalization factors
            max_data_size = max(current_data_sizes.values()) if current_data_sizes else 1
            if max_data_size <= 0:
                max_data_size = 1
            
            for client_id in participating_clients:
                # 1. Current performance weight (primary)
                current_loss_imp = current_loss_improvements.get(client_id, 0.0)
                if current_loss_imp > 0:
                    performance_weight = min(1.0, current_loss_imp * 5.0)
                else:
                    performance_weight = max(0.1, 0.5 + current_loss_imp * 0.5)
                
                # 2. Historical reliability weight
                if client_id in self.client_reliability and len(self.client_reliability[client_id]) > 0:
                    client_qualities = self.client_reliability[client_id]
                    valid_qualities = [q for q in client_qualities if not (np.isnan(q) or np.isinf(q))]
                    
                    if valid_qualities:
                        # Recent performance matters more
                        recent_qualities = valid_qualities[-5:]  # Last 5 rounds
                        avg_quality = np.mean(recent_qualities)
                        reliability_weight = max(0.1, min(1.0, avg_quality))
                    else:
                        reliability_weight = 0.3
                else:
                    reliability_weight = 0.3  # Default for new clients
                
                # 3. Data size weight (normalized)
                data_weight = current_data_sizes.get(client_id, 1) / max_data_size
                
                # 4. Consistency weight (based on loss improvement history)
                if client_id in self.client_loss_history and len(self.client_loss_history[client_id]) > 2:
                    loss_history = self.client_loss_history[client_id]
                    valid_losses = [l for l in loss_history if not (np.isnan(l) or np.isinf(l))]
                    
                    if len(valid_losses) > 2:
                        # Prefer clients with consistent positive improvements
                        positive_ratio = sum(1 for l in valid_losses if l > 0) / len(valid_losses)
                        consistency_weight = positive_ratio
                    else:
                        consistency_weight = 0.5
                else:
                    consistency_weight = 0.5
                
                # Combined weight - prioritize current performance and reliability
                final_weight = (0.4 * performance_weight + 
                               0.3 * reliability_weight + 
                               0.2 * consistency_weight + 
                               0.1 * data_weight)
                
                # Ensure valid weight
                if np.isnan(final_weight) or np.isinf(final_weight) or final_weight <= 0:
                    final_weight = 0.1  # Minimum weight
                
                weights[client_id] = final_weight
            
            # Normalize weights
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
        """Get reliability statistics for all clients"""
        stats = {}
        for client_id, qualities in self.client_reliability.items():
            try:
                valid_qualities = [q for q in qualities if not (np.isnan(q) or np.isinf(q))]
                valid_losses = [l for l in self.client_loss_history[client_id] 
                               if not (np.isnan(l) or np.isinf(l))]
                
                if valid_qualities:
                    avg_quality = np.mean(valid_qualities)
                    std_quality = np.std(valid_qualities)
                    participation = self.client_participation[client_id]
                    
                    # Adam-specific reliability score
                    recent_qualities = valid_qualities[-5:] if len(valid_qualities) >= 5 else valid_qualities
                    recent_avg = np.mean(recent_qualities)
                    
                    # Factor in loss improvement consistency
                    if valid_losses:
                        positive_improvements = sum(1 for l in valid_losses if l > 0)
                        improvement_ratio = positive_improvements / len(valid_losses)
                        reliability_score = recent_avg * (1 + improvement_ratio) * np.log(1 + participation)
                    else:
                        reliability_score = recent_avg * np.log(1 + participation)
                    
                    # Validate stats
                    if np.isnan(avg_quality) or np.isinf(avg_quality):
                        avg_quality = 0.3
                    if np.isnan(std_quality) or np.isinf(std_quality):
                        std_quality = 0.0
                    if np.isnan(reliability_score) or np.isinf(reliability_score):
                        reliability_score = 0.3
                    
                    stats[client_id] = {
                        'avg_quality': avg_quality,
                        'recent_quality': recent_avg,
                        'std_quality': std_quality,
                        'participation_count': participation,
                        'reliability_score': reliability_score,
                        'improvement_ratio': improvement_ratio if valid_losses else 0.0
                    }
                else:
                    stats[client_id] = {
                        'avg_quality': 0.3,
                        'recent_quality': 0.3,
                        'std_quality': 0.0,
                        'participation_count': self.client_participation[client_id],
                        'reliability_score': 0.3,
                        'improvement_ratio': 0.0
                    }
            except Exception as e:
                print(f"⚠️  Error computing stats for client {client_id}: {e}")
                stats[client_id] = {
                    'avg_quality': 0.3,
                    'recent_quality': 0.3,
                    'std_quality': 0.0,
                    'participation_count': 1,
                    'reliability_score': 0.3,
                    'improvement_ratio': 0.0
                }
        
        return stats
    
    def _remove_oldest_memory(self):
        """Remove oldest memory to maintain size limit"""
        try:
            if len(self.update_embeddings) > 0:
                # Remove from all lists
                self.update_embeddings.pop(0)
                self.quality_metrics.pop(0)
                removed_client = self.client_ids.pop(0)
                self.round_numbers.pop(0)
                self.data_sizes.pop(0)
                self.loss_improvements.pop(0)
                
                # Rebuild FAISS index
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                if self.update_embeddings:
                    embeddings_array = np.array(self.update_embeddings)
                    self.index.add(embeddings_array)
                
                print(f"Removed oldest memory from client {removed_client}")
        except Exception as e:
            print(f"⚠️  Error removing oldest memory: {e}")
    
    def save_memory_bank(self, filepath: str):
        """Save memory bank to disk"""
        try:
            data = {
                'update_embeddings': self.update_embeddings,
                'quality_metrics': self.quality_metrics,
                'client_ids': self.client_ids,
                'round_numbers': self.round_numbers,
                'data_sizes': self.data_sizes,
                'loss_improvements': self.loss_improvements,
                'client_reliability': dict(self.client_reliability),
                'client_participation': dict(self.client_participation),
                'client_loss_history': dict(self.client_loss_history),
                'embedding_dim': self.embedding_dim,
                'max_memories': self.max_memories,
                'similarity_threshold': self.similarity_threshold
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"Adam-optimized memory bank saved to {filepath}")
        except Exception as e:
            print(f"⚠️  Error saving memory bank: {e}")
    
    def load_memory_bank(self, filepath: str):
        """Load memory bank from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.update_embeddings = data['update_embeddings']
            self.quality_metrics = data['quality_metrics']
            self.client_ids = data['client_ids']
            self.round_numbers = data['round_numbers']
            self.data_sizes = data['data_sizes']
            self.loss_improvements = data.get('loss_improvements', [])
            self.client_reliability = defaultdict(list, data['client_reliability'])
            self.client_participation = defaultdict(int, data['client_participation'])
            self.client_loss_history = defaultdict(list, data.get('client_loss_history', {}))
            
            # Rebuild FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            if self.update_embeddings:
                embeddings_array = np.array(self.update_embeddings)
                self.index.add(embeddings_array)
            
            print(f"Adam-optimized memory bank loaded from {filepath}")
        except Exception as e:
            print(f"⚠️  Error loading memory bank: {e}")

# Removed all SGD-related code - No more optimizer type switching
# Renamed methods - compute_gradient_embedding → compute_update_embedding
# Adam-focused quality computation - Prioritizes loss improvement (80% weight)
# Enhanced tracking - Added client_loss_history for consistency tracking
# Adam-specific embeddings - Features tailored for parameter updates
# Improved weighting - Considers recent performance, consistency, and reliability
