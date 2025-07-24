import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy

class LocalUpdateWithGradients:
    """Adam-optimized LocalUpdate that computes parameter updates and loss improvements"""
    
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
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                batch_size=max(1, int(len(idxs_val)/10)), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                               batch_size=max(1, int(len(idxs_test)/10)), shuffle=False)

        return trainloader, validloader, testloader

    def compute_loss(self, model):
        """Compute current loss on local training data"""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)
        
        return total_loss / total_samples if total_samples > 0 else 0.0

    def update_weights(self, model, global_round):
        """
        Train model with Adam optimizer and return weights, loss improvement, and parameter updates
        
        Returns:
            final_weights: Updated model weights
            loss_improvement: How much loss improved during training
            param_updates: Parameter updates (final - initial weights)
            training_info: Additional training statistics
        """
        # Store initial state
        initial_weights = copy.deepcopy(model.state_dict())
        initial_loss = self.compute_loss(model)
        
        # Set mode to train
        model.train()
        epoch_losses = []
        
        # Adam optimizer with recommended settings for federated learning
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4
        )
        
        # Training loop
        for epoch in range(self.args.local_ep):
            batch_losses = []
            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability with Adam
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()

                batch_losses.append(loss.item())
                
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, epoch, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
            
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            
            # Early stopping for Adam if loss becomes very small (prevents overfitting)
            if epoch_loss < 0.001:
                if self.args.verbose:
                    print(f'Early stopping at epoch {epoch} due to very low loss: {epoch_loss:.6f}')
                break
        
        # Compute final loss and loss improvement
        final_loss = self.compute_loss(model)
        loss_improvement = initial_loss - final_loss
        
        # Get final weights
        final_weights = model.state_dict()
        
        # Compute parameter updates (what Adam actually applied)
        param_updates = {}
        for key in initial_weights.keys():
            param_updates[key] = final_weights[key] - initial_weights[key]
        
        # Additional training statistics
        training_info = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'epoch_losses': epoch_losses,
            'num_epochs_trained': len(epoch_losses),
            'avg_epoch_loss': sum(epoch_losses) / len(epoch_losses),
            'total_samples': len(self.trainloader.dataset)
        }
        
        if self.args.verbose:
            print(f'Client training completed: Loss {initial_loss:.4f} → {final_loss:.4f} '
                  f'(improvement: {loss_improvement:.4f})')
        
        return final_weights, loss_improvement, param_updates, training_info

    def update_weights_with_validation(self, model, global_round):
        """
        Enhanced training with validation monitoring for better quality assessment
        
        Returns same as update_weights but with validation metrics
        """
        # Store initial state
        initial_weights = copy.deepcopy(model.state_dict())
        initial_loss = self.compute_loss(model)
        initial_val_loss = self.compute_validation_loss(model)
        
        # Set mode to train
        model.train()
        epoch_losses = []
        val_losses = []
        
        # Adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 3  # Early stopping patience
        
        # Training loop with validation
        for epoch in range(self.args.local_ep):
            # Training phase
            model.train()
            batch_losses = []
            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_losses.append(loss.item())
            
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            
            # Validation phase
            val_loss = self.compute_validation_loss(model)
            val_losses.append(val_loss)
            
            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if self.args.verbose:
                    print(f'Early stopping at epoch {epoch} due to validation loss plateau')
                break
            
            if self.args.verbose and epoch % 2 == 0:
                print(f'Epoch {epoch}: Train Loss {epoch_loss:.4f}, Val Loss {val_loss:.4f}')
        
        # Compute final metrics
        final_loss = self.compute_loss(model)
        final_val_loss = self.compute_validation_loss(model)
        
        loss_improvement = initial_loss - final_loss
        val_improvement = initial_val_loss - final_val_loss
        
        # Get final weights and compute updates
        final_weights = model.state_dict()
        param_updates = {}
        for key in initial_weights.keys():
            param_updates[key] = final_weights[key] - initial_weights[key]
        
        # Enhanced training statistics
        training_info = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'initial_val_loss': initial_val_loss,
            'final_val_loss': final_val_loss,
            'val_improvement': val_improvement,
            'epoch_losses': epoch_losses,
            'val_losses': val_losses,
            'num_epochs_trained': len(epoch_losses),
            'avg_epoch_loss': sum(epoch_losses) / len(epoch_losses),
            'best_val_loss': best_val_loss,
            'total_samples': len(self.trainloader.dataset)
        }
        
        if self.args.verbose:
            print(f'Client training: Loss {initial_loss:.4f}→{final_loss:.4f} '
                  f'Val {initial_val_loss:.4f}→{final_val_loss:.4f} '
                  f'(improvements: {loss_improvement:.4f}, {val_improvement:.4f})')
        
        return final_weights, loss_improvement, param_updates, training_info

    def compute_validation_loss(self, model):
        """Compute validation loss"""
        if len(self.validloader) == 0:
            return self.compute_loss(model)  # Fallback to training loss
            
        model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in self.validloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)
        
        return total_loss / total_samples if total_samples > 0 else 0.0

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

    def get_data_size(self):
        """Return the size of local training dataset"""
        return len(self.trainloader.dataset)

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

# Fixed Adam parameters with recommended settings for FL
# param_updates[key] = final_weights[key] - initial_weights[key]
#    -Returns actual parameter changes that Adam applied
# Computes initial and final loss on training data
#    -loss_improvement = initial_loss - final_loss ---> what memory bank uses for quality assesment
# 
#
#
#
