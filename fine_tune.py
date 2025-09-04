import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from copy import deepcopy
import random

# Import modules


class TransferLearningExperiment:
    """
    Experiment class for fine-tuning a model trained on 15 GHz data to work with other data.
    Shows performance vs. amount of training data using validation set for evaluation.
    """
    
    def __init__(self, pretrained_model_path, config_15ghz, config_8ghz):

        self.pretrained_model_path = pretrained_model_path
        self.config_15ghz = config_15ghz
        self.config_8ghz = config_8ghz
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the model architecture
        self.base_model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
        
        # Load pretrained weights
        print(f"Loading pretrained model from {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path, map_location=self.device,weights_only=False)
        
        # Handle both checkpoint and state_dict formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.base_model.load_state_dict(checkpoint)
        
        # Setup datasets
        self._setup_datasets()
        
    def _setup_datasets(self):
        
        self.rss_processor_8ghz = RSSMapProcessor(
            image_path=self.config_8ghz['rss_image_path'],
            bs_pixel_coords=self.config_8ghz['bs_pixel_coords'],
            bs_real_coords=self.config_8ghz['bs_real_coords'],
            image_width_meters=self.config_8ghz['image_width_meters']
        )
        
        self.train_dataset_8ghz = GlobalNormalizedDataset(
            self.config_8ghz['smomp_file'],
            self.config_8ghz['accurate_file'],
            self.config_8ghz['user_positions_file'],
            self.rss_processor_8ghz,
            split='train',
            train_ratio=0.7,
            val_ratio=0.3
        )
        
        self.val_dataset_8ghz = GlobalNormalizedDataset(
            self.config_8ghz['smomp_file'],
            self.config_8ghz['accurate_file'],
            self.config_8ghz['user_positions_file'],
            self.rss_processor_8ghz,
            split='val',
            train_ratio=0.7,
            val_ratio=0.3
        )
        
        print(f"8 GHz dataset sizes - Train: {len(self.train_dataset_8ghz)}, "
              f"Val: {len(self.val_dataset_8ghz)}")
    
    def fine_tune_model(self, n_samples, epochs=50, lr=5e-4, freeze_encoder=False):
        """
        Fine-tune the pretrained model on a subset of 8 GHz data.
        
        Args:
            n_samples: Number of training samples to use
            epochs: Number of fine-tuning epochs
            lr: Learning rate for fine-tuning (lower than original training)
            freeze_encoder: Whether to freeze encoder layers (for limited data)
            
        Returns:
            fine_tuned_model: The fine-tuned model
            train_history: Training loss history
            val_history: Validation loss history
            best_val_nmse: Best validation NMSE achieved
        """
        
        # Create a copy of the base model
        model = deepcopy(self.base_model)
        model = model.to(self.device)
        
        # Optionally freeze encoder layers for very limited data scenarios
        if freeze_encoder and n_samples < 100:
            print(f"Freezing encoder layers (n_samples={n_samples} < 100)")
            for name, param in model.named_parameters():
                if 'enc' in name and 'decoder' not in name:
                    param.requires_grad = False
        
        # Create subset of training data
        if n_samples < len(self.train_dataset_8ghz):
            # Randomly sample indices
            all_indices = list(range(len(self.train_dataset_8ghz)))
            random.shuffle(all_indices)
            subset_indices = all_indices[:n_samples]
            train_subset = Subset(self.train_dataset_8ghz, subset_indices)
        else:
            train_subset = self.train_dataset_8ghz
            n_samples = len(self.train_dataset_8ghz)
        
        print(f"\nFine-tuning with {n_samples} samples for {epochs} epochs...")
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=min(32, n_samples), shuffle=True)
        val_loader = DataLoader(self.val_dataset_8ghz, batch_size=32, shuffle=False)
        
        # Setup optimizer with lower learning rate for fine-tuning
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        # Learning rate scheduler with warmup for stability
        warmup_epochs = min(5, epochs // 4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs)
        
        # Loss function
        criterion = PhysicsInformedLoss(alpha=0.01)
        
        # Training history
        train_history = []
        val_history = []
        best_val_nmse = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Warmup phase
            if epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * (epoch + 1) / warmup_epochs
            
            # Training phase
            model.train()
            train_loss = 0
            train_nmse = 0
            
            for batch_idx, (smomp, accurate, rss) in enumerate(train_loader):
                smomp = smomp.to(self.device)
                accurate = accurate.to(self.device)
                rss = rss.to(self.device)
                
                optimizer.zero_grad()
                pred = model(smomp, rss)
                loss, nmse, _ = criterion(pred, accurate, rss)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_nmse += nmse.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_train_nmse = train_nmse / len(train_loader)
            train_history.append(avg_train_nmse)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_nmse = 0
            
            with torch.no_grad():
                for smomp, accurate, rss in val_loader:
                    smomp = smomp.to(self.device)
                    accurate = accurate.to(self.device)
                    rss = rss.to(self.device)
                    
                    pred = model(smomp, rss)
                    loss, nmse, _ = criterion(pred, accurate, rss)
                    
                    val_loss += loss.item()
                    val_nmse += nmse.item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_nmse = val_nmse / len(val_loader)
            val_history.append(avg_val_nmse)
            
            # Save best model
            if avg_val_nmse < best_val_nmse:
                best_val_nmse = avg_val_nmse
                best_model_state = deepcopy(model.state_dict())
            
            # Update scheduler after warmup
            if epoch >= warmup_epochs:
                scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'Epoch {epoch+1}/{epochs}: Train NMSE={avg_train_nmse:.4f}, Val NMSE={avg_val_nmse:.4f}')
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, train_history, val_history, best_val_nmse
    
    def evaluate_model_on_validation(self, model):
        """
        Evaluate a model on the validation set.
        
        Args:
            model: Model to evaluate
            
        Returns:
            nmse: Average NMSE on validation set
            nmse_db: NMSE in dB
        """
        model = model.to(self.device)
        model.eval()
        
        val_loader = DataLoader(self.val_dataset_8ghz, batch_size=32, shuffle=False)
        
        total_nmse = 0
        n_samples = 0
        
        with torch.no_grad():
            for smomp, accurate, rss in val_loader:
                smomp = smomp.to(self.device)
                accurate = accurate.to(self.device)
                rss = rss.to(self.device)
                
                pred = model(smomp, rss)
                
                # Calculate NMSE
                n_channels = pred.shape[1] // 2
                pred_complex = torch.complex(pred[:, :n_channels], pred[:, n_channels:])
                accurate_complex = torch.complex(accurate[:, :n_channels], accurate[:, n_channels:])
                
                mse = torch.mean(torch.abs(pred_complex - accurate_complex) ** 2)
                target_power = torch.mean(torch.abs(accurate_complex) ** 2)
                nmse = mse / target_power
                
                total_nmse += nmse.item() * smomp.shape[0]
                n_samples += smomp.shape[0]
        
        avg_nmse = total_nmse / n_samples
        nmse_db = 10 * np.log10(avg_nmse)
        
        return avg_nmse, nmse_db
    
    def run_experiment(self, sample_sizes=None, epochs_per_size=50):
        """
        Run the complete transfer learning experiment using validation set for evaluation.
        
        Args:
            sample_sizes: List of training sample sizes to test
            epochs_per_size: Number of epochs for each sample size
            
        Returns:
            results: Dictionary with experiment results
        """
        
        results = {
            'sample_sizes': [],
            'nmse_values': [],
            'nmse_db_values': [],
            'best_val_nmse_during_training': [],
            'baseline_nmse': None,
            'baseline_nmse_db': None,
            'train_histories': [],
            'val_histories': []
        }
        
        # Evaluate baseline (pretrained model without fine-tuning)
        print("\nEvaluating baseline (15 GHz model on 8 GHz validation data)...")
        baseline_nmse, baseline_nmse_db = self.evaluate_model_on_validation(self.base_model)
        results['baseline_nmse'] = baseline_nmse
        results['baseline_nmse_db'] = baseline_nmse_db
        print(f"Baseline Validation NMSE: {baseline_nmse:.6f} ({baseline_nmse_db:.2f} dB)")
        
        # Run fine-tuning experiments
        for n_samples in sample_sizes:
            print(f"\n{'='*60}")
            print(f"Fine-tuning with {n_samples} training samples")
            print(f"{'='*60}")
            
            # Adjust epochs based on sample size
            epochs = epochs_per_size
            if n_samples < 50:
                epochs = min(100, epochs * 2)  # More epochs for very small datasets
            
            # Adjust learning rate based on sample size
            lr = 5e-4 if n_samples >= 100 else 1e-4
            
            # Fine-tune model
            fine_tuned_model, train_hist, val_hist, best_val_nmse = self.fine_tune_model(
                n_samples, 
                epochs=epochs, 
                lr=lr,
                freeze_encoder=(n_samples < 50)
            )
            
            # Final evaluation on validation set
            nmse, nmse_db = self.evaluate_model_on_validation(fine_tuned_model)
            
            results['sample_sizes'].append(n_samples)
            results['nmse_values'].append(nmse)
            results['nmse_db_values'].append(nmse_db)
            results['best_val_nmse_during_training'].append(best_val_nmse)
            results['train_histories'].append(train_hist)
            results['val_histories'].append(val_hist)
            
            print(f"Final Validation NMSE: {nmse:.6f} ({nmse_db:.2f} dB)")
            print(f"Improvement over baseline: {baseline_nmse_db - nmse_db:.2f} dB")
            
            # Save best model for each sample size (as state_dict only for compatibility)
            save_path = f'fine_tuned_8ghz_{n_samples}_samples.pth'
            torch.save(fine_tuned_model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        return results