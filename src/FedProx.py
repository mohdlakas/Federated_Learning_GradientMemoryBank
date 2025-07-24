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
from utils import get_dataset, average_weights, exp_details

class LocalUpdateFedProx:
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = torch.nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # ✅ Convert to list of integers first
        idxs = [int(i) for i in idxs]
        
        # split indexes for train, validation, and test (80, 10, 10)
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

    def update_weights(self, model, global_round, mu=0.01):
        """
        FedProx local update with proximal term
        mu: proximal term coefficient (default 0.01)
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        # Store global model parameters for proximal term
        global_model_params = {name: param.clone().detach() 
                             for name, param in model.named_parameters()}

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                
                # Standard loss
                loss = self.criterion(log_probs, labels)
                
                # Add proximal term: mu/2 * ||w - w_global||^2
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    proximal_term += torch.norm(param - global_model_params[name]) ** 2
                
                total_loss = loss + (mu / 2.0) * proximal_term
                
                total_loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), total_loss.item()))
                
                batch_loss.append(total_loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

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

    # Initialize tracking
    train_loss, train_accuracy = [], []
    print_every = 2

    # TRAINING LOOP
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        
        print(f'\n | Global Training Round : {epoch+1} |\n')
        global_model.train()
        
        # Sample clients
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local training
        for idx in idxs_users:
            local_model = LocalUpdateFedProx(args=args, dataset=train_dataset,
                                           idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, mu=0.01)
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # Global aggregation
        global_weights = average_weights(local_weights)
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
    
    file_name = f'{objects_save_path}/{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_fedprox.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))