#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args, seed=42):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    
    Args:
        args: Command line arguments
        seed: Random seed for reproducible data splits
    """
    
    if args.dataset == 'cifar':
        data_dir = '/Users/ml/Desktop/gradient_memory_bank_FL/data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # ✅ SAMPLE TRAINING DATA WITH SEED FOR REPRODUCIBILITY
        if args.iid:
            # Sample IID user data from CIFAR10
            user_groups = cifar_iid(train_dataset, args.num_users, seed=seed)
        else:
            # Sample Non-IID user data from CIFAR10
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users, seed=seed)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '/Users/ml/Desktop/gradient_memory_bank_FL/data/MNIST/' 
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))]))
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))]))
        else:  # fmnist
            data_dir = '/Users/ml/Desktop/gradient_memory_bank_FL/data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))]))
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,), (0.3081,))]))

        # ✅ SAMPLE TRAINING DATA WITH SEED FOR REPRODUCIBILITY
        if args.iid:
            # Sample IID user data from MNIST/Fashion-MNIST
            user_groups = mnist_iid(train_dataset, args.num_users, seed=seed)
        else:
            # Sample Non-IID user data from MNIST/Fashion-MNIST
            if args.unequal:
                # Chose unequal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users, seed=seed)
            else:
                # Chose equal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users, seed=seed)

    # ✅ PRINT VERIFICATION INFO
    print(f"🔧 Dataset: {args.dataset}, Distribution: {'IID' if args.iid else 'Non-IID'}")
    print(f"📊 Total users: {args.num_users}, Seed: {seed}")
    print(f"📈 Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Quick verification of splits
    total_assigned = sum(len(user_groups[i]) for i in range(args.num_users))
    print(f"✅ Data split verification: {total_assigned}/{len(train_dataset)} samples assigned")

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_intelligent(w, intelligent_weights, participating_clients):
    """
    Returns the weighted average of the weights using intelligent weights from memory bank.
    
    Args:
        w: List of client model weights
        intelligent_weights: Dict of {client_id: weight} from memory bank
        participating_clients: List of client IDs that participated this round
    """
    try:
        # Initialize with the first client's weights
        w_avg = copy.deepcopy(w[0])
        
        # Zero out all weights first
        for key in w_avg.keys():
            w_avg[key] = torch.zeros_like(w_avg[key])
        
        # Apply intelligent weighted averaging
        total_weight = 0.0
        for i, client_id in enumerate(participating_clients):
            client_weight = intelligent_weights.get(client_id, 1.0)
            
            # Validate weight
            if client_weight <= 0 or torch.isnan(torch.tensor(client_weight)) or torch.isinf(torch.tensor(client_weight)):
                client_weight = 1.0
            
            total_weight += client_weight
            
            # Add weighted client contribution
            for key in w_avg.keys():
                w_avg[key] += w[i][key] * client_weight
        
        # Normalize by total weight
        if total_weight > 0:
            for key in w_avg.keys():
                w_avg[key] = torch.div(w_avg[key], total_weight)
        else:
            # Fallback to standard averaging
            print("⚠️  Total weight is 0, falling back to standard averaging")
            return average_weights(w)
        
        return w_avg
        
    except Exception as e:
        print(f"⚠️  Error in intelligent averaging: {e}")
        print("Falling back to standard averaging")
        return average_weights(w)


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return