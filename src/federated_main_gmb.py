#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
import random
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

# Add these imports for Gradient Memory Bank
from gmb import GradientMemoryBank
from update_gmb import LocalUpdateWithGradients

from memory_bank_visualizer import MemoryBankVisualizer
from performance_comparator import PerformanceComparator

# ✅ REPRODUCIBILITY SEEDING - MUST BE AT THE TOP
EXPERIMENT_SEED = 42

def set_all_seeds(seed=EXPERIMENT_SEED):
    """Set all random seeds for complete reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔧 [GMB] All random seeds set to {seed} for reproducibility")

# Set seeds immediately
set_all_seeds(EXPERIMENT_SEED)

def average_weights_intelligent(w_list, weight_dict, client_ids):
    """
    Aggregate model weights using intelligent weighting from memory bank
    """
    w_avg = copy.deepcopy(w_list[0])
    
    # Initialize with zeros
    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])
    
    # Weighted aggregation
    for i, client_id in enumerate(client_ids):
        weight = weight_dict[client_id]
        for k in w_avg.keys():
            w_avg[k] += weight * w_list[i][k]
    
    return w_avg

def select_intelligent_clients(memory_bank, num_clients, total_clients, epoch, diversity_factor=0.9):
    """
    Select clients intelligently based on memory bank reliability with accurate strategy reporting
    
    Args:
        memory_bank: GradientMemoryBank instance
        num_clients: Number of clients to select (typically 5)
        total_clients: Total number of available clients (typically 100)
        epoch: Current training round
        diversity_factor: Fraction of clients selected randomly for diversity (0.6 = 60% random, 40% intelligent)
    
    Returns:
        selected: Array of selected client IDs
        strategy: Detailed strategy description
    """
    if epoch == 0:
        # First round: random selection (no history yet)
        selected = np.random.choice(range(total_clients), num_clients, replace=False)
        return selected, "random_cold_start"
    
    # Get reliability stats
    reliability_stats = memory_bank.get_client_reliability_stats()
    
    if not reliability_stats or len(reliability_stats) < num_clients:
        # Fallback to random if insufficient history
        selected = np.random.choice(range(total_clients), num_clients, replace=False)
        return selected, "random_fallback"
    
    # Calculate intelligent vs random split
    reliable_count = max(1, int((1.0 - diversity_factor) * num_clients))  # 40% = 2 clients
    random_count = num_clients - reliable_count  # 60% = 3 clients
    
    # Sort clients by reliability score (descending)
    sorted_clients = sorted(reliability_stats.items(), 
                          key=lambda x: x[1]['reliability_score'], reverse=True)
    
    selected = []
    
    # 🎯 SELECT TOP RELIABLE CLIENTS (intelligent selection)
    reliable_clients_selected = []
    for i in range(min(reliable_count, len(sorted_clients))):
        client_id = sorted_clients[i][0]
        selected.append(client_id)
        reliable_clients_selected.append(client_id)
    
    # 🎲 ADD RANDOM CLIENTS FOR DIVERSITY
    random_clients_selected = []
    if random_count > 0:
        all_clients = set(range(total_clients))
        available_clients = list(all_clients - set(selected))
        if available_clients:
            additional = np.random.choice(available_clients, 
                                        min(random_count, len(available_clients)), 
                                        replace=False)
            selected.extend(additional)
            random_clients_selected.extend(additional)
    
    # 📊 GENERATE ACCURATE STRATEGY DESCRIPTION
    actual_reliable = len(reliable_clients_selected)
    actual_random = len(random_clients_selected)
    
    # Strategy naming based on actual composition
    if actual_random > actual_reliable:
        strategy = f"mixed_random_dominant_{actual_reliable}i_{actual_random}r"
    elif actual_reliable > actual_random:
        strategy = f"mixed_intelligent_dominant_{actual_reliable}i_{actual_random}r"
    else:
        strategy = f"mixed_balanced_{actual_reliable}i_{actual_random}r"
    
    # Add diversity factor info
    strategy += f"_div{int(diversity_factor*100)}"
    
    return np.array(selected[:num_clients]), strategy

def analyze_client_selection_breakdown(client_selection_log):
    """
    Analyze the true breakdown of client selection strategies
    """
    total_rounds = len(client_selection_log)
    total_clients_selected = 0
    intelligent_clients_selected = 0
    random_clients_selected = 0
    
    strategy_breakdown = {}
    
    for log in client_selection_log:
        strategy = log['strategy']
        num_clients_in_round = len(log['clients'])
        total_clients_selected += num_clients_in_round
        
        # Count strategies
        if strategy not in strategy_breakdown:
            strategy_breakdown[strategy] = 0
        strategy_breakdown[strategy] += 1
        
        # Extract intelligent vs random counts from strategy name
        if 'mixed_' in strategy:
            # Parse strategy like "mixed_random_dominant_2i_3r_div60"
            parts = strategy.split('_')
            for part in parts:
                if part.endswith('i'):  # intelligent clients
                    intelligent_clients_selected += int(part[:-1])
                elif part.endswith('r'):  # random clients
                    random_clients_selected += int(part[:-1])
        elif strategy == 'random_cold_start':
            random_clients_selected += num_clients_in_round
        elif strategy == 'random_fallback':
            random_clients_selected += num_clients_in_round
    
    return {
        'total_rounds': total_rounds,
        'total_clients_selected': total_clients_selected,
        'intelligent_clients_selected': intelligent_clients_selected,
        'random_clients_selected': random_clients_selected,
        'intelligent_percentage': (intelligent_clients_selected / total_clients_selected * 100) if total_clients_selected > 0 else 0,
        'random_percentage': (random_clients_selected / total_clients_selected * 100) if total_clients_selected > 0 else 0,
        'strategy_breakdown': strategy_breakdown
    }

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('/Users/ml/Desktop/gradient_memory_bank_FL/logs')

    args = args_parser()
    exp_details(args)

    # GPU setup
    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # ✅ LOAD DATASET WITH SEED FOR REPRODUCIBLE DATA SPLITS
    train_dataset, test_dataset, user_groups = get_dataset(args, seed=EXPERIMENT_SEED)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer perceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device
    global_model.to(device)
    global_model.train()
    print(global_model)

    # ✅ PRINT MODEL INITIALIZATION INFO FOR VERIFICATION
    total_params = sum(p.numel() for p in global_model.parameters())
    init_weight_sum = sum(p.sum().item() for p in global_model.parameters())
    print(f"🔍 [GMB] Model initialization verification:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Initial weight sum: {init_weight_sum:.6f}")

    # Initialize Gradient Memory Bank
    memory_bank = GradientMemoryBank(
        embedding_dim=512,
        max_memories=1000,
        similarity_threshold=0.7,
        device=device
    )
    print(f"🧠 [GMB] Memory Bank initialized with {memory_bank.max_memories} max memories")

    # Track data sizes for each client
    client_data_sizes = {i: len(user_groups[i]) for i in range(args.num_users)}

    # Initialize tracking lists
    train_loss, train_accuracy = [], []
    client_selection_log = []  # Track how clients are selected
    print_every = 2

    # ✅ VERIFY CLIENT SAMPLING REPRODUCIBILITY
    print(f"🔍 [GMB] Client sampling verification (intelligent vs random):")

    # TRAINING LOOP with Memory Bank Integration
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        client_gradients = {}  # Store gradients for memory bank
        loss_improvements = {}  # Track loss improvements
        
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # Set model to training mode
        global_model.train()
        
        # ✅ INTELLIGENT CLIENT SELECTION using Memory Bank
        m = max(int(args.frac * args.num_users), 1)
        idxs_users, selection_strategy = select_intelligent_clients(
            memory_bank, m, args.num_users, epoch, diversity_factor=0.9
        )
        
        client_selection_log.append({
            'epoch': epoch,
            'strategy': selection_strategy,
            'clients': list(idxs_users)
        })
        
        # Print selection info for first few rounds
        if epoch < 5:
            print(f"  Round {epoch+1}: {selection_strategy} selection: {sorted(idxs_users)}")
            
            # Show reliability stats for selected clients
            if epoch > 0:
                reliability_stats = memory_bank.get_client_reliability_stats()
                if reliability_stats:
                    print("    🧠 Selected clients reliability:")
                    for client_id in sorted(idxs_users):
                        if client_id in reliability_stats:
                            stats = reliability_stats[client_id]
                            print(f"      Client {client_id}: Reliability={stats['reliability_score']:.3f}, "
                                  f"Quality={stats['avg_quality']:.3f}, Participation={stats['participation_count']}")

        # Store previous global loss for computing improvements
        if epoch > 0:
            global_model.eval()
            with torch.no_grad():
                _, prev_global_loss = test_inference(args, global_model, test_dataset)
        else:
            prev_global_loss = float('inf')

        # LOCAL TRAINING on selected clients
        for idx in idxs_users:
            local_model = LocalUpdateWithGradients(args=args, dataset=train_dataset,
                                                 idxs=user_groups[idx], logger=logger)
            
            # Get weights, loss, and gradients
            w, loss, gradients = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            client_gradients[idx] = gradients
            
            # Compute loss improvement (previous loss - current loss)
            loss_improvements[idx] = max(0.0, prev_global_loss - loss)

        # INTELLIGENT AGGREGATION using Memory Bank
        if epoch > 0:  # Skip first round (no history yet)
            # Get intelligent weights from memory bank
            current_data_sizes = {idx: client_data_sizes[idx] for idx in idxs_users}
            intelligent_weights = memory_bank.compute_client_weights(
                participating_clients=idxs_users,
                current_gradients=client_gradients,
                current_data_sizes=current_data_sizes
            )
            
            # Apply intelligent weights to aggregation
            global_weights = average_weights_intelligent(local_weights, intelligent_weights, idxs_users)
            
            # Debug aggregation weights
            if epoch < 5:
                print(f"    📊 Aggregation weights: {[f'{idx}:{intelligent_weights[idx]:.3f}' for idx in sorted(idxs_users)]}")
        else:
            # Use standard FedAvg for first round
            global_weights = average_weights(local_weights)
        
        global_model.load_state_dict(global_weights)

        # ADD GRADIENTS TO MEMORY BANK
        for idx in idxs_users:
            memory_bank.add_gradient_memory(
                client_id=idx,
                gradients=client_gradients[idx],
                loss_improvement=loss_improvements[idx],
                data_size=client_data_sizes[idx],
                round_num=epoch
            )

        # # Debug quality computation for first few rounds
        # if epoch < 3:
        #     print(f"    🔍 Quality debug for round {epoch+1}:")
        #     for idx in list(idxs_users)[:3]:  # Show first 3 clients
        #         loss_imp = loss_improvements[idx]
        #         if idx in client_gradients:
        #             try:
        #                 # Safe gradient norm calculation
        #                 gradients = client_gradients[idx]
        #                 if isinstance(gradients, flist) and len(gradients) > 0:
        #                     # Check if gradients are tensors
        #                     if hasattr(gradients[0], 'flatten'):
        #                         grad_norm = torch.norm(torch.cat([g.flatten() for g in gradients])).item()
        #                     else:
        #                         grad_norm = "N/A (not tensor)"
        #                 else:
        #                     grad_norm = "N/A (empty/invalid)"
        #             except Exception as e:
        #                 grad_norm = f"Error: {str(e)[:50]}"
        #         else:
        #             grad_norm = "N/A (no gradients)"
                
        #         print(f"      Client {idx}: Loss improvement={loss_imp:.4f}, Grad norm={grad_norm}")

        # Calculate average training loss for this round
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # EVALUATION: Test the global model
        global_model.eval()
        with torch.no_grad():
            test_acc_round, test_loss_round = test_inference(args, global_model, test_dataset)
            train_accuracy.append(test_acc_round)

        # Enhanced logging with memory bank stats
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Test Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            
            # Print memory bank statistics
            reliability_stats = memory_bank.get_client_reliability_stats()
            if reliability_stats:
                print("Top 5 Most Reliable Clients:")
                sorted_clients = sorted(reliability_stats.items(), 
                                      key=lambda x: x[1]['reliability_score'], reverse=True)[:5]
                for client_id, stats in sorted_clients:
                    print(f"  Client {client_id}: Reliability={stats['reliability_score']:.3f}, "
                          f"Avg Quality={stats['avg_quality']:.3f}, "
                          f"Participation={stats['participation_count']}")
                
                # Show selection strategy distribution
                recent_strategies = [log['strategy'] for log in client_selection_log[-10:]]
                strategy_counts = {s: recent_strategies.count(s) for s in set(recent_strategies)}
                print(f"  Recent selection strategies: {strategy_counts}")

    # FINAL EVALUATION
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Final Test Accuracy: {:.2f}%".format(100*test_acc))
    print("|---- Test Accuracy (last round): {:.2f}%".format(100*train_accuracy[-1]))

    # CREATE SAVE DIRECTORIES
    objects_save_path = "/Users/ml/Desktop/gradient_memory_bank_FL/save/objects"
    images_save_path = "/Users/ml/Desktop/gradient_memory_bank_FL/save/images"
    
    os.makedirs(objects_save_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)

    # ✅ ENHANCED FILENAME WITH SEED FOR TRACKING
    file_name = f'{objects_save_path}/{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_seed[{EXPERIMENT_SEED}]_gmb_intelligent.pkl'
    
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy, client_selection_log], f)

    # Save memory bank at the end
    memory_bank_path = f'{objects_save_path}/memory_bank_{args.dataset}_{args.model}_{args.epochs}_seed[{EXPERIMENT_SEED}]_intelligent.pkl'
    memory_bank.save_memory_bank(memory_bank_path)
    print(f'Memory bank saved to: {memory_bank_path}')

    # VISUALIZATION AND ANALYSIS
    print("\n" + "="*60)
    print("GENERATING MEMORY BANK VISUALIZATIONS AND ANALYSIS")
    print("="*60)

    # Create visualizer
    visualizer = MemoryBankVisualizer(memory_bank, images_save_path)

    # Generate all visualizations
    print("1. Generating client reliability evolution plots...")
    visualizer.plot_client_reliability_evolution()

    print("2. Generating gradient embedding space visualization...")
    visualizer.plot_gradient_embedding_space()

    print("3. Generating comprehensive memory bank statistics...")
    visualizer.plot_memory_bank_statistics()

    print("4. Generating memory bank analysis report...")
    visualizer.generate_memory_bank_report()

    # PERFORMANCE COMPARISON
    print("\n5. Setting up performance comparison...")
    comparator = PerformanceComparator(images_save_path)

    # Add current experiment (Memory Bank enhanced)
    comparator.add_experiment(
        name="GMB Intelligent",
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        metadata={
            'method': 'Gradient Memory Bank with Intelligent Client Selection',
            'embedding_dim': 512,
            'max_memories': 1000,
            'similarity_threshold': 0.7,
            'seed': EXPERIMENT_SEED
        }
    )

    # Try to load baseline results for comparison
    baseline_file = f'{objects_save_path}/{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_seed[{EXPERIMENT_SEED}]_fedavg.pkl'
    try:
        comparator.load_experiment_from_pickle(
            name="Standard FedAvg",
            filepath=baseline_file,
            metadata={'method': 'Standard FedAvg', 'seed': EXPERIMENT_SEED}
        )
        
        # Try to load previous GMB results too
        prev_gmb_file = f'{objects_save_path}/{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_seed[{EXPERIMENT_SEED}]_gmb.pkl'
        try:
            comparator.load_experiment_from_pickle(
                name="GMB Random Selection",
                filepath=prev_gmb_file,
                metadata={'method': 'GMB with Random Client Selection', 'seed': EXPERIMENT_SEED}
            )
        except FileNotFoundError:
            print("Previous GMB results not found")
        
        print("6. Generating performance comparison plots...")
        comparator.plot_performance_comparison()
        
        print("7. Generating detailed loss trajectory analysis...")
        comparator.create_detailed_loss_trajectory_plot()
        
    except FileNotFoundError:
        print(f"Baseline results not found at {baseline_file}")
        print("Run seeded FedAvg first to enable comparison")

    print(f"\nAll visualizations saved to: {images_save_path}")
    print("Analysis complete!")

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    print(f'📊 [GMB Intelligent] Results saved with seed {EXPERIMENT_SEED} for reproducibility')

    # PLOTTING - Combined plot only
    plt.figure(figsize=(15, 6))

    # Plot Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(range(len(train_loss)), train_loss, color='r', linewidth=2, label='GMB Intelligent Loss')
    plt.title('GMB Intelligent Training Loss vs Communication Rounds')
    plt.ylabel('Training Loss')
    plt.xlabel('Communication Rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot Accuracy curve
    plt.subplot(1, 3, 2)
    plt.plot(range(len(train_accuracy)), [acc*100 for acc in train_accuracy], color='b', linewidth=2, label='GMB Intelligent Accuracy')
    plt.title('GMB Intelligent Test Accuracy vs Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.xlabel('Communication Rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot Client Selection Strategy Distribution
    plt.subplot(1, 3, 3)
    strategies = [log['strategy'] for log in client_selection_log]
    strategy_counts = {s: strategies.count(s) for s in set(strategies)}
    plt.pie(strategy_counts.values(), labels=strategy_counts.keys(), autopct='%1.1f%%')
    plt.title('Client Selection Strategy Distribution')

    plt.tight_layout()
    
    # ✅ ENHANCED PLOT FILENAME WITH METHOD AND SEED
    plot_filename = f'{images_save_path}/gmb_intelligent_{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_seed[{EXPERIMENT_SEED}].png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

    print(f'\n📈 GMB Intelligent plot saved to: {plot_filename}')
    print(f'💾 GMB Intelligent data saved to: {file_name}')
    print(f'🔧 Experiment seed: {EXPERIMENT_SEED} (consistent across all FL methods)')
    
    # ENHANCED MEMORY BANK ANALYSIS
    print("\n" + "="*60)
    print("INTELLIGENT GMB FINAL SUMMARY")
    print("="*60)
    
    # Print final memory bank statistics
    reliability_stats = memory_bank.get_client_reliability_stats()
    if reliability_stats:
        print(f"Total unique clients in memory bank: {len(reliability_stats)}")
        print(f"Total gradient memories stored: {len(memory_bank.gradient_embeddings)}")
        print(f"Average gradient quality: {np.mean(memory_bank.quality_metrics):.4f}")
        
        # CLIENT SELECTION STRATEGY DETAILED ANALYSIS
        selection_analysis = analyze_client_selection_breakdown(client_selection_log)
        
        print(f"\n📊 DETAILED CLIENT SELECTION ANALYSIS:")
        print(f"Total rounds: {selection_analysis['total_rounds']}")
        print(f"Total client selections: {selection_analysis['total_clients_selected']}")
        print(f"Intelligent selections: {selection_analysis['intelligent_clients_selected']} ({selection_analysis['intelligent_percentage']:.1f}%)")
        print(f"Random selections: {selection_analysis['random_clients_selected']} ({selection_analysis['random_percentage']:.1f}%)")
        
        print(f"\nStrategy Distribution:")
        for strategy, count in selection_analysis['strategy_breakdown'].items():
            percentage = (count / selection_analysis['total_rounds']) * 100
            print(f"  {strategy}: {count} rounds ({percentage:.1f}%)")
        
        # Most selected clients
        all_selected_clients = []
        for log in client_selection_log:
            all_selected_clients.extend(log['clients'])
        
        from collections import Counter
        client_selection_counts = Counter(all_selected_clients)
        most_selected = client_selection_counts.most_common(10)
        
        print(f"\nTop 10 Most Selected Clients:")
        for i, (client_id, selection_count) in enumerate(most_selected, 1):
            if client_id in reliability_stats:
                stats = reliability_stats[client_id]
                print(f"{i:2d}. Client {client_id:3d}: Selected {selection_count} times, "
                      f"Reliability={stats['reliability_score']:.3f}, "
                      f"Quality={stats['avg_quality']:.3f}")
        
        # Most reliable clients
        top_reliable = sorted(reliability_stats.items(), 
                            key=lambda x: x[1]['reliability_score'], reverse=True)[:10]
        print(f"\nTop 10 Most Reliable Clients:")
        for i, (client_id, stats) in enumerate(top_reliable, 1):
            selection_count = client_selection_counts.get(client_id, 0)
            print(f"{i:2d}. Client {client_id:3d}: Reliability={stats['reliability_score']:.3f}, "
                  f"Quality={stats['avg_quality']:.3f}, Selected {selection_count} times")
    
    print(f"\nIntelligent Memory Bank Federated Learning Complete!")
    print(f"Results saved with '_gmb_intelligent' suffix and seed {EXPERIMENT_SEED}")
