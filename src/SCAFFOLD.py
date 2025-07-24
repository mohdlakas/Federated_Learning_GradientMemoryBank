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
from tensorboardX import SummaryWriter

from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, exp_details

class LocalUpdateSCAFFOLD:
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = torch.nn.NLLLoss().to(self.device)

# In the LocalUpdateSCAFFOLD class, update the train_val_test method:

    def train_val_test(self, dataset, idxs):
        # ✅ Add this line
        idxs = [int(i) for i in idxs]
        
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(dataset, idxs_train),
            batch_size=self.args.local_bs, shuffle=True)
        validloader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(dataset, idxs_val),
            batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(dataset, idxs_test),
            batch_size=int(len(idxs_test)/10), shuffle=False)

        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, c_global, c_local):
        model.train()
        epoch_loss = []

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # Store initial model
        initial_model = {name: param.clone().detach() for name, param in model.named_parameters()}

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                # ✅ SIMPLIFIED: Scaled control variate correction
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Scale down the correction to prevent explosion
                        correction = 0.1 * (c_global[name] - c_local[name])
                        param.grad.data = param.grad.data + correction
                        
                        # ✅ Add gradient clipping
                        torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # ✅ SIMPLIFIED: Conservative control variate update  
        c_new_local = {}
        local_steps = len(self.trainloader) * self.args.local_ep
        
        for name, param in model.named_parameters():
            # Use smaller update step
            param_diff = initial_model[name] - param.data
            update_step = param_diff / (local_steps * self.args.lr * 10)  # Divide by 10 for stability
            
            c_new_local[name] = c_local[name] - c_global[name] + update_step
            
            # ✅ Clamp control variates to prevent explosion
            c_new_local[name] = torch.clamp(c_new_local[name], -1.0, 1.0)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), c_new_local

def scaffold_aggregate(local_weights, local_control_variates, global_control_variate, num_selected, total_clients, lr):
    """
    SCAFFOLD aggregation with control variate updates
    """
    # Standard weight averaging
    w_avg = copy.deepcopy(local_weights[0])
    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])
    
    for w in local_weights:
        for k in w_avg.keys():
            w_avg[k] += w[k] / len(local_weights)

    # Update global control variate
    c_delta_avg = {k: torch.zeros_like(v) for k, v in global_control_variate.items()}
    
    for c_local in local_control_variates:
        for k in c_delta_avg.keys():
            c_delta_avg[k] += (c_local[k] - global_control_variate[k]) / num_selected

    # Update global control variate
    for k in global_control_variate.keys():
        global_control_variate[k] += c_delta_avg[k] * (num_selected / total_clients)

    return w_avg, global_control_variate

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

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)

    global_model.to(device)
    global_model.train()
    print(global_model)

    # Initialize SCAFFOLD control variates
    global_control_variate = {name: torch.zeros_like(param) 
                             for name, param in global_model.named_parameters()}
    
    client_control_variates = {}
    for i in range(args.num_users):
        client_control_variates[i] = {name: torch.zeros_like(param) 
                                    for name, param in global_model.named_parameters()}

    # Initialize tracking
    train_loss, train_accuracy = [], []
    print_every = 2

    # TRAINING LOOP
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_control_variates = []
        
        print(f'\n | Global Training Round : {epoch+1} |\n')
        global_model.train()
        
        # Sample clients
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local training with SCAFFOLD
        for idx in idxs_users:
            local_model = LocalUpdateSCAFFOLD(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=logger)
            w, loss, c_new_local = local_model.update_weights(
                model=copy.deepcopy(global_model), 
                global_round=epoch,
                c_global=global_control_variate,
                c_local=client_control_variates[idx])
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_control_variates.append(c_new_local)
            
            # Update client's control variate
            client_control_variates[idx] = c_new_local

        # SCAFFOLD aggregation
        global_weights, global_control_variate = scaffold_aggregate(
            local_weights, local_control_variates, global_control_variate,
            len(idxs_users), args.num_users, args.lr)
        
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Test inference
        global_model.eval()
        with torch.no_grad():
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            train_accuracy.append(test_acc)

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Test Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Final results
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Final Test Accuracy: {:.2f}%".format(100*test_acc))

    # Save results
    objects_save_path = "/Users/ml/Desktop/gradient_memory_bank_FL/save/objects"
    os.makedirs(objects_save_path, exist_ok=True)
    
    file_name = f'{objects_save_path}/{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_scaffold.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))