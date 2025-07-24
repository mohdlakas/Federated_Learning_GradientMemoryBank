import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy

class LocalUpdateWithGradients:
    """Extended LocalUpdate that also computes and returns gradients"""
    
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """Returns train, validation and test dataloaders for a given dataset and user indexes."""
        # Split indexes for train, validation, and test (90%, 5%, 5%)
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_val = idxs[int(0.9*len(idxs)):int(0.95*len(idxs))]
        idxs_test = idxs[int(0.95*len(idxs)):]
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                batch_size=max(1, int(len(idxs_val)/10)), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=max(1, int(len(idxs_test)/10)), shuffle=False)
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                        batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)

        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        """Train model and return weights, loss, and gradients"""
        # Set mode to train model
        model.train()
        epoch_loss = []
        
        # Store initial weights for gradient computation
        initial_weights = copy.deepcopy(model.state_dict())

        # Set optimizer for the local client
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                # # Add gradient clipping for Adam
                # if self.args.optimizer == 'adam':
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # # Early stopping if loss gets too low (prevents overfitting)
            # if self.args.optimizer == 'adam' and epoch_loss[-1] < 0.01:
            #     if self.args.verbose:
            #         print(f'Early stopping at epoch {iter} due to low loss: {epoch_loss[-1]:.6f}')
            #     break
        # Compute gradients (difference between final and initial weights)
        final_weights = model.state_dict()
        gradients = {}
        
        for key in initial_weights.keys():
            weight_diff = initial_weights[key] - final_weights[key]
            if self.args.optimizer == 'sgd':
                gradients[key] = weight_diff / self.args.lr
            else:  # Adam - use raw weight difference
                gradients[key] = weight_diff
        
        return final_weights, sum(epoch_loss) / len(epoch_loss), gradients

    def inference(self, model):
        """Returns the inference accuracy and loss."""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)