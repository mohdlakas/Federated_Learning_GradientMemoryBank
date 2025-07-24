import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MemoryBankVisualizer:
    """
    Comprehensive visualization tools for Gradient Memory Bank analysis
    """
    
    def __init__(self, memory_bank, save_path: str):
        self.memory_bank = memory_bank
        self.save_path = save_path
        
    def plot_client_reliability_evolution(self):
        """Plot how client reliability changes over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Client Quality Over Time
        client_data = {}
        for i, (client_id, round_num, quality) in enumerate(
            zip(self.memory_bank.client_ids, self.memory_bank.round_numbers, self.memory_bank.quality_metrics)):
            if client_id not in client_data:
                client_data[client_id] = {'rounds': [], 'qualities': []}
            client_data[client_id]['rounds'].append(round_num)
            client_data[client_id]['qualities'].append(quality)
        
        # Plot top 10 most active clients
        sorted_clients = sorted(client_data.items(), key=lambda x: len(x[1]['rounds']), reverse=True)[:10]
        
        ax1 = axes[0, 0]
        for client_id, data in sorted_clients:
            ax1.plot(data['rounds'], data['qualities'], marker='o', alpha=0.7, label=f'Client {client_id}')
        ax1.set_xlabel('Communication Round')
        ax1.set_ylabel('Gradient Quality')
        ax1.set_title('Client Quality Evolution Over Time')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Quality Distribution by Round
        ax2 = axes[0, 1]
        rounds = np.array(self.memory_bank.round_numbers)
        qualities = np.array(self.memory_bank.quality_metrics)
        
        round_quality_data = {}
        for round_num, quality in zip(rounds, qualities):
            if round_num not in round_quality_data:
                round_quality_data[round_num] = []
            round_quality_data[round_num].append(quality)
        
        round_means = [np.mean(round_quality_data[r]) for r in sorted(round_quality_data.keys())]
        round_stds = [np.std(round_quality_data[r]) for r in sorted(round_quality_data.keys())]
        round_nums = sorted(round_quality_data.keys())
        
        ax2.errorbar(round_nums, round_means, yerr=round_stds, marker='o', capsize=5)
        ax2.set_xlabel('Communication Round')
        ax2.set_ylabel('Average Quality Score')
        ax2.set_title('Average Quality Evolution with Error Bars')
        ax2.grid(True, alpha=0.3)
        
        # 3. Client Participation Frequency
        ax3 = axes[1, 0]
        reliability_stats = self.memory_bank.get_client_reliability_stats()
        if reliability_stats:
            client_ids = list(reliability_stats.keys())
            participation_counts = [reliability_stats[c]['participation_count'] for c in client_ids]
            
            ax3.bar(range(len(client_ids)), participation_counts, alpha=0.7)
            ax3.set_xlabel('Client ID')
            ax3.set_ylabel('Participation Count')
            ax3.set_title('Client Participation Frequency')
            ax3.set_xticks(range(0, len(client_ids), max(1, len(client_ids)//10)))
        
        # 4. Quality vs Participation Scatter
        ax4 = axes[1, 1]
        if reliability_stats:
            avg_qualities = [reliability_stats[c]['avg_quality'] for c in client_ids]
            reliability_scores = [reliability_stats[c]['reliability_score'] for c in client_ids]
            
            scatter = ax4.scatter(participation_counts, avg_qualities, 
                                c=reliability_scores, cmap='viridis', alpha=0.7)
            ax4.set_xlabel('Participation Count')
            ax4.set_ylabel('Average Quality')
            ax4.set_title('Quality vs Participation (Color = Reliability)')
            plt.colorbar(scatter, ax=ax4, label='Reliability Score')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/memory_bank_client_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_gradient_embedding_space(self):
        """Visualize gradient embeddings in 2D space using t-SNE"""
        if len(self.memory_bank.gradient_embeddings) < 10:
            print("Not enough gradient embeddings for visualization")
            return
            
        # Convert embeddings to numpy array
        embeddings = np.array(self.memory_bank.gradient_embeddings)
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create interactive plot with Plotly
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('Colored by Quality', 'Colored by Client ID'))
        
        # Plot 1: Color by quality
        fig.add_trace(
            go.Scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.memory_bank.quality_metrics,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Quality Score", x=0.45)
                ),
                text=[f'Client: {cid}<br>Round: {rnd}<br>Quality: {qual:.3f}' 
                      for cid, rnd, qual in zip(self.memory_bank.client_ids, 
                                               self.memory_bank.round_numbers,
                                               self.memory_bank.quality_metrics)],
                hovertemplate='%{text}<extra></extra>',
                name='Quality'
            ),
            row=1, col=1
        )
        
        # Plot 2: Color by client ID
        fig.add_trace(
            go.Scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.memory_bank.client_ids,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Client ID", x=1.0)
                ),
                text=[f'Client: {cid}<br>Round: {rnd}<br>Quality: {qual:.3f}' 
                      for cid, rnd, qual in zip(self.memory_bank.client_ids, 
                                               self.memory_bank.round_numbers,
                                               self.memory_bank.quality_metrics)],
                hovertemplate='%{text}<extra></extra>',
                name='Client ID'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Gradient Embedding Space Visualization (t-SNE)',
            showlegend=False,
            height=600,
            width=1200
        )
        
        fig.write_html(f'{self.save_path}/gradient_embedding_space.html')
        print(f"Interactive embedding visualization saved to {self.save_path}/gradient_embedding_space.html")
        
    def plot_memory_bank_statistics(self):
        """Plot comprehensive memory bank statistics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Quality Distribution Histogram
        ax1 = axes[0, 0]
        ax1.hist(self.memory_bank.quality_metrics, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Quality Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Gradient Quality Scores')
        ax1.axvline(np.mean(self.memory_bank.quality_metrics), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.memory_bank.quality_metrics):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Data Size vs Quality
        ax2 = axes[0, 1]
        scatter = ax2.scatter(self.memory_bank.data_sizes, self.memory_bank.quality_metrics, 
                            c=self.memory_bank.round_numbers, cmap='plasma', alpha=0.7)
        ax2.set_xlabel('Client Data Size')
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Data Size vs Quality (Color = Round)')
        plt.colorbar(scatter, ax=ax2, label='Communication Round')
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory Bank Growth Over Time
        ax3 = axes[0, 2]
        rounds = np.array(self.memory_bank.round_numbers)
        unique_rounds = np.unique(rounds)
        memory_counts = [np.sum(rounds <= r) for r in unique_rounds]
        
        ax3.plot(unique_rounds, memory_counts, marker='o', linewidth=2, markersize=6)
        ax3.set_xlabel('Communication Round')
        ax3.set_ylabel('Total Memories Stored')
        ax3.set_title('Memory Bank Growth Over Time')
        ax3.grid(True, alpha=0.3)
        
        # 4. Top Clients by Reliability
        ax4 = axes[1, 0]
        reliability_stats = self.memory_bank.get_client_reliability_stats()
        if reliability_stats:
            top_clients = sorted(reliability_stats.items(), 
                               key=lambda x: x[1]['reliability_score'], reverse=True)[:15]
            
            client_ids = [f"C{x[0]}" for x in top_clients]
            reliability_scores = [x[1]['reliability_score'] for x in top_clients]
            
            bars = ax4.bar(client_ids, reliability_scores, alpha=0.7, color='lightgreen', edgecolor='black')
            ax4.set_xlabel('Client ID')
            ax4.set_ylabel('Reliability Score')
            ax4.set_title('Top 15 Clients by Reliability')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, reliability_scores):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Quality Evolution Trend
        ax5 = axes[1, 1]
        if len(unique_rounds) > 1:
            round_avg_quality = [np.mean([q for q, r in zip(self.memory_bank.quality_metrics, rounds) if r == round_num]) 
                               for round_num in unique_rounds]
            
            ax5.plot(unique_rounds, round_avg_quality, marker='o', linewidth=2, markersize=6, color='orange')
            ax5.set_xlabel('Communication Round')
            ax5.set_ylabel('Average Quality Score')
            ax5.set_title('Quality Evolution Trend')
            
            # Add trend line
            z = np.polyfit(unique_rounds, round_avg_quality, 1)
            p = np.poly1d(z)
            ax5.plot(unique_rounds, p(unique_rounds), "--", alpha=0.7, color='red',
                    label=f'Trend: {z[0]:.4f}x + {z[1]:.3f}')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Client Diversity Analysis
        ax6 = axes[1, 2]
        if reliability_stats:
            quality_stds = [reliability_stats[c]['std_quality'] for c in reliability_stats.keys()]
            avg_qualities = [reliability_stats[c]['avg_quality'] for c in reliability_stats.keys()]
            
            ax6.scatter(avg_qualities, quality_stds, alpha=0.7, color='purple', s=50)
            ax6.set_xlabel('Average Quality')
            ax6.set_ylabel('Quality Standard Deviation')
            ax6.set_title('Client Consistency vs Quality')
            ax6.grid(True, alpha=0.3)
            
            # Add quadrant lines
            ax6.axhline(np.median(quality_stds), color='gray', linestyle='--', alpha=0.5)
            ax6.axvline(np.median(avg_qualities), color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/memory_bank_comprehensive_stats.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_memory_bank_report(self):
        """Generate a comprehensive text report of memory bank statistics"""
        stats = self.memory_bank.get_client_reliability_stats()
        
        report = f"""
{'='*60}
GRADIENT MEMORY BANK ANALYSIS REPORT
{'='*60}

GENERAL STATISTICS:
- Total memories stored: {len(self.memory_bank.gradient_embeddings)}
- Communication rounds covered: {min(self.memory_bank.round_numbers)} - {max(self.memory_bank.round_numbers)}
- Unique clients participated: {len(set(self.memory_bank.client_ids))}
- Average quality score: {np.mean(self.memory_bank.quality_metrics):.4f} ± {np.std(self.memory_bank.quality_metrics):.4f}

QUALITY METRICS:
- Highest quality score: {max(self.memory_bank.quality_metrics):.4f}
- Lowest quality score: {min(self.memory_bank.quality_metrics):.4f}
- Quality score range: {max(self.memory_bank.quality_metrics) - min(self.memory_bank.quality_metrics):.4f}

CLIENT RELIABILITY ANALYSIS:
"""
        
        if stats:
            # Top 10 most reliable clients
            top_reliable = sorted(stats.items(), key=lambda x: x[1]['reliability_score'], reverse=True)[:10]
            report += "\nTOP 10 MOST RELIABLE CLIENTS:\n"
            report += f"{'Client ID':<10} {'Reliability':<12} {'Avg Quality':<12} {'Participation':<12}\n"
            report += "-" * 50 + "\n"
            
            for client_id, client_stats in top_reliable:
                report += f"{client_id:<10} {client_stats['reliability_score']:<12.4f} "
                report += f"{client_stats['avg_quality']:<12.4f} {client_stats['participation_count']:<12}\n"
            
            # Most active clients
            most_active = sorted(stats.items(), key=lambda x: x[1]['participation_count'], reverse=True)[:10]
            report += "\nTOP 10 MOST ACTIVE CLIENTS:\n"
            report += f"{'Client ID':<10} {'Participation':<12} {'Avg Quality':<12} {'Reliability':<12}\n"
            report += "-" * 50 + "\n"
            
            for client_id, client_stats in most_active:
                report += f"{client_id:<10} {client_stats['participation_count']:<12} "
                report += f"{client_stats['avg_quality']:<12.4f} {client_stats['reliability_score']:<12.4f}\n"
        
        report += f"\n{'='*60}\n"
        
        # Save report
        with open(f'{self.save_path}/memory_bank_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        print(f"Detailed report saved to {self.save_path}/memory_bank_report.txt")