import matplotlib.pyplot as plt
import numpy as np
import pickle
from typing import Dict, List, Tuple
import seaborn as sns

class PerformanceComparator:
    """
    Compare performance between standard FedAvg and Memory Bank enhanced FL
    """
    
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.experiments = {}
        
    def add_experiment(self, name: str, train_loss: List[float], 
                      train_accuracy: List[float], metadata: Dict = None):
        """Add experimental results for comparison"""
        self.experiments[name] = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'metadata': metadata or {}
        }
        print(f"Added experiment: {name}")
        
    def load_experiment_from_pickle(self, name: str, filepath: str, metadata: Dict = None):
        """Load experimental results from pickle file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                train_loss, train_accuracy = data
                self.add_experiment(name, train_loss, train_accuracy, metadata)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
    def plot_performance_comparison(self):
        """Create comprehensive performance comparison plots"""
        if len(self.experiments) < 2:
            print("Need at least 2 experiments for comparison")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Training Loss Comparison
        ax1 = axes[0, 0]
        for name, data in self.experiments.items():
            rounds = range(len(data['train_loss']))
            ax1.plot(rounds, data['train_loss'], marker='o', linewidth=2, 
                    markersize=4, label=name, alpha=0.8)
        
        ax1.set_xlabel('Communication Rounds')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Test Accuracy Comparison
        ax2 = axes[0, 1]
        for name, data in self.experiments.items():
            rounds = range(len(data['train_accuracy']))
            accuracy_percent = [acc * 100 for acc in data['train_accuracy']]
            ax2.plot(rounds, accuracy_percent, marker='s', linewidth=2, 
                    markersize=4, label=name, alpha=0.8)
        
        ax2.set_xlabel('Communication Rounds')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Test Accuracy Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Convergence Rate Analysis
        ax3 = axes[1, 0]
        convergence_data = {}
        
        for name, data in self.experiments.items():
            # Calculate convergence rate (rounds to reach 95% of final accuracy)
            final_acc = data['train_accuracy'][-1]
            target_acc = 0.95 * final_acc
            
            convergence_round = len(data['train_accuracy'])  # Default to last round
            for i, acc in enumerate(data['train_accuracy']):
                if acc >= target_acc:
                    convergence_round = i
                    break
            
            convergence_data[name] = {
                'convergence_round': convergence_round,
                'final_accuracy': final_acc * 100,
                'improvement_rate': (final_acc - data['train_accuracy'][0]) / len(data['train_accuracy'])
            }
        
        # Bar plot for convergence rounds
        names = list(convergence_data.keys())
        conv_rounds = [convergence_data[name]['convergence_round'] for name in names]
        bars = ax3.bar(names, conv_rounds, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'][:len(names)])
        
        ax3.set_ylabel('Rounds to 95% of Final Accuracy')
        ax3.set_title('Convergence Rate Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rounds in zip(bars, conv_rounds):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{rounds}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Performance Improvement Analysis
        ax4 = axes[1, 1]
        
        # Calculate performance metrics
        metrics_data = []
        for name, data in self.experiments.items():
            final_loss = data['train_loss'][-1]
            final_acc = data['train_accuracy'][-1] * 100
            loss_reduction = (data['train_loss'][0] - final_loss) / data['train_loss'][0] * 100
            acc_improvement = (final_acc - data['train_accuracy'][0] * 100)
            
            metrics_data.append({
                'name': name,
                'final_accuracy': final_acc,
                'loss_reduction': loss_reduction,
                'accuracy_improvement': acc_improvement
            })
        
        # Create grouped bar chart
        x = np.arange(len(metrics_data))
        width = 0.35
        
        final_accs = [m['final_accuracy'] for m in metrics_data]
        acc_improvements = [m['accuracy_improvement'] for m in metrics_data]
        
        bars1 = ax4.bar(x - width/2, final_accs, width, label='Final Accuracy (%)', alpha=0.8)
        bars2 = ax4.bar(x + width/2, acc_improvements, width, label='Accuracy Improvement (%)', alpha=0.8)
        
        ax4.set_xlabel('Experiments')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Final Performance Metrics')
        ax4.set_xticks(x)
        ax4.set_xticklabels([m['name'] for m in metrics_data], rotation=45)
        ax4.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print numerical comparison
        self.print_numerical_comparison(convergence_data, metrics_data)
        
    def print_numerical_comparison(self, convergence_data: Dict, metrics_data: List[Dict]):
        """Print detailed numerical comparison"""
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n{'Experiment':<20} {'Final Acc (%)':<15} {'Conv. Rounds':<15} {'Loss Reduction (%)':<18}")
        print("-" * 70)
        
        for name in self.experiments.keys():
            conv_data = convergence_data[name]
            metric_data = next(m for m in metrics_data if m['name'] == name)
            
            print(f"{name:<20} {conv_data['final_accuracy']:<15.2f} "
                  f"{conv_data['convergence_round']:<15} {metric_data['loss_reduction']:<18.2f}")
        
        # Calculate improvements
        if len(self.experiments) == 2:
            names = list(self.experiments.keys())
            baseline_name = names[0]  # Assume first is baseline
            enhanced_name = names[1]  # Assume second is enhanced
            
            baseline_conv = convergence_data[baseline_name]
            enhanced_conv = convergence_data[enhanced_name]
            
            acc_improvement = enhanced_conv['final_accuracy'] - baseline_conv['final_accuracy']
            conv_improvement = baseline_conv['convergence_round'] - enhanced_conv['convergence_round']
            
            print(f"\n{'='*60}")
            print("MEMORY BANK IMPROVEMENTS:")
            print(f"{'='*60}")
            print(f"Accuracy improvement: {acc_improvement:+.2f}%")
            print(f"Convergence speed improvement: {conv_improvement:+} rounds")
            print(f"Relative accuracy gain: {(acc_improvement/baseline_conv['final_accuracy']*100):+.2f}%")
            
    def create_detailed_loss_trajectory_plot(self):
        """Create detailed loss trajectory analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Loss trajectories with confidence intervals (if multiple runs available)
        ax1 = axes[0]
        for name, data in self.experiments.items():
            rounds = range(len(data['train_loss']))
            ax1.plot(rounds, data['train_loss'], marker='o', linewidth=2, 
                    markersize=3, label=name, alpha=0.8)
            
            # Add trend line
            z = np.polyfit(rounds, data['train_loss'], 2)  # Polynomial fit
            p = np.poly1d(z)
            ax1.plot(rounds, p(rounds), '--', alpha=0.6, linewidth=1)
        
        ax1.set_xlabel('Communication Rounds')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Loss Trajectories with Trend Lines')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # 2. Loss reduction rate
        ax2 = axes[1]
        for name, data in self.experiments.items():
            loss_reductions = []
            for i in range(1, len(data['train_loss'])):
                reduction = (data['train_loss'][i-1] - data['train_loss'][i]) / data['train_loss'][i-1]
                loss_reductions.append(reduction)
            
            rounds = range(1, len(data['train_loss']))
            ax2.plot(rounds, loss_reductions, marker='s', linewidth=2, 
                    markersize=3, label=name, alpha=0.8)
        
        ax2.set_xlabel('Communication Rounds')
        ax2.set_ylabel('Loss Reduction Rate')
        ax2.set_title('Loss Reduction Rate per Round')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/detailed_loss_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()