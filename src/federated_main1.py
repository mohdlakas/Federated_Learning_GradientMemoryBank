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
    print(f"🔧 [FedAvg] All random seeds set to {seed} for reproducibility")

# Set seeds immediately
set_all_seeds(EXPERIMENT_SEED)

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
    print(f"🔍 Model initialization verification:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Initial weight sum: {init_weight_sum:.6f}")

    # Initialize tracking lists
    train_loss, train_accuracy = [], []
    print_every = 2

    # ✅ VERIFY CLIENT SAMPLING REPRODUCIBILITY
    print(f"🔍 Client sampling verification (first few rounds):")

    # TRAINING LOOP
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # Set model to training mode
        global_model.train()
        
        # ✅ REPRODUCIBLE CLIENT SAMPLING (numpy seed already set)
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # Print first few rounds for verification
        if epoch < 3:
            print(f"  Round {epoch+1}: Selected clients {sorted(idxs_users)}")

        # LOCAL TRAINING on selected clients
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # GLOBAL AGGREGATION (Simple FedAvg)
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # Calculate average training loss for this round
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # EVALUATION: Test the global model
        global_model.eval()
        with torch.no_grad():
            test_acc_round, test_loss_round = test_inference(args, global_model, test_dataset)
            train_accuracy.append(test_acc_round)

        # Print progress every few rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Test Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # FINAL EVALUATION
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Final Test Accuracy: {:.2f}%".format(100*test_acc))
    print("|---- Test Accuracy (last round): {:.2f}%".format(100*train_accuracy[-1]))

    # ✅ ENHANCED FILENAME WITH SEED FOR TRACKING
    objects_save_path = "/Users/ml/Desktop/gradient_memory_bank_FL/save/objects"
    images_save_path = "/Users/ml/Desktop/gradient_memory_bank_FL/save/images"
    
    os.makedirs(objects_save_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)

    # SAVE TRAINING DATA with seed in filename
    file_name = f'{objects_save_path}/{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_seed[{EXPERIMENT_SEED}]_fedavg.pkl'
    
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    print(f'📊 [FedAvg] Results saved with seed {EXPERIMENT_SEED} for reproducibility')

# PLOTTING
plt.figure(figsize=(12, 6))

# Plot Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(len(train_loss)), train_loss, color='r', linewidth=2, label='FedAvg Training Loss')
plt.title('FedAvg Training Loss vs Communication Rounds')
plt.ylabel('Training Loss')
plt.xlabel('Communication Rounds')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(range(len(train_accuracy)), [acc*100 for acc in train_accuracy], color='b', linewidth=2, label='FedAvg Test Accuracy')
plt.title('FedAvg Test Accuracy vs Communication Rounds')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Communication Rounds')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

# ✅ ENHANCED PLOT FILENAME WITH METHOD AND SEED
plot_filename = f'{images_save_path}/fedavg_{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_seed[{EXPERIMENT_SEED}].png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

print(f'\n📈 FedAvg plot saved to: {plot_filename}')
print(f'💾 FedAvg data saved to: {file_name}')
print(f'🔧 Experiment seed: {EXPERIMENT_SEED} (use same seed for all FL methods)')
