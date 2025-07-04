#!/usr/bin/env python3
"""
Optimized ML Modeling with Overfitting Prevention
================================================

This module provides enhanced optimization methods with comprehensive
overfitting prevention strategies for fluid flow modeling.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class OptimizedFluidModeler:
    """
    Enhanced fluid modeling with advanced optimization and overfitting prevention.
    """
    
    def __init__(self, X_data, Y_data, train_ratio=0.7, val_ratio=0.15):
        """
        Initialize the optimized fluid modeler.
        
        Parameters:
        -----------
        X_data : np.ndarray
            Input features (pressure, water holdup, gas holdup)
        Y_data : np.ndarray
            Target variables (water velocity, gas velocity)
        train_ratio : float
            Ratio of training data
        val_ratio : float
            Ratio of validation data
        """
        self.X_data = X_data
        self.Y_data = Y_data
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Data preprocessing
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        # Split data
        self._prepare_data()
        
        # Model parameters
        self.n_features = X_data.shape[0] // 3
        self.n_outputs = Y_data.shape[0] // 2
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_model_state = None
        
    def _prepare_data(self):
        """Prepare and split the data."""
        # Reshape data for time series
        X_reshaped = self.X_data.T  # (time_steps, features)
        Y_reshaped = self.Y_data.T  # (time_steps, outputs)
        
        # Split data
        n_samples = X_reshaped.shape[0]
        train_size = int(n_samples * self.train_ratio)
        val_size = int(n_samples * self.val_ratio)
        
        # Train/Val/Test split
        X_train = X_reshaped[:train_size]
        Y_train = Y_reshaped[:train_size]
        
        X_val = X_reshaped[train_size:train_size + val_size]
        Y_val = Y_reshaped[train_size:train_size + val_size]
        
        X_test = X_reshaped[train_size + val_size:]
        Y_test = Y_reshaped[train_size + val_size:]
        
        # Scale data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        Y_train_scaled = self.scaler_Y.fit_transform(Y_train)
        
        X_val_scaled = self.scaler_X.transform(X_val)
        Y_val_scaled = self.scaler_Y.transform(Y_val)
        
        X_test_scaled = self.scaler_X.transform(X_test)
        Y_test_scaled = self.scaler_Y.transform(Y_test)
        
        # Convert to PyTorch tensors
        self.X_train = torch.tensor(X_train_scaled, dtype=torch.float64)
        self.Y_train = torch.tensor(Y_train_scaled, dtype=torch.float64)
        self.X_val = torch.tensor(X_val_scaled, dtype=torch.float64)
        self.Y_val = torch.tensor(Y_val_scaled, dtype=torch.float64)
        self.X_test = torch.tensor(X_test_scaled, dtype=torch.float64)
        self.Y_test = torch.tensor(Y_test_scaled, dtype=torch.float64)
        
        print(f"Data shapes:")
        print(f"  Train: X={self.X_train.shape}, Y={self.Y_train.shape}")
        print(f"  Val: X={self.X_val.shape}, Y={self.Y_val.shape}")
        print(f"  Test: X={self.X_test.shape}, Y={self.Y_test.shape}")
    
    def custom_loss(self, pred, target, A, B, lambda_l1=1e-5, lambda_l2=1e-4):
        """
        Custom loss function with L1 and L2 regularization.
        
        Parameters:
        -----------
        pred : torch.Tensor
            Model predictions
        target : torch.Tensor
            True targets
        A, B : torch.Tensor
            Model parameters
        lambda_l1, lambda_l2 : float
            Regularization coefficients
            
        Returns:
        --------
        torch.Tensor : Total loss
        """
        mse_loss = nn.MSELoss()(pred, target)
        l1_reg = lambda_l1 * (torch.norm(A, p=1) + torch.norm(B, p=1))
        l2_reg = lambda_l2 * (torch.norm(A, p=2)**2 + torch.norm(B, p=2)**2)
        return mse_loss + l1_reg + l2_reg
    
    def train_model(self, max_epochs=10000, patience=200, lr=0.01, 
                   weight_decay=1e-4, lambda_l1=1e-5, lambda_l2=1e-4):
        """
        Train the model with comprehensive overfitting prevention.
        
        Parameters:
        -----------
        max_epochs : int
            Maximum training epochs
        patience : int
            Early stopping patience
        lr : float
            Learning rate
        weight_decay : float
            Weight decay for optimizer
        lambda_l1, lambda_l2 : float
            Regularization coefficients
        """
        # Initialize parameters with proper scaling
        A = torch.randn(2*self.n_features, 2*self.n_features, 
                       dtype=torch.float64, requires_grad=True) * 0.01
        B = torch.randn(2*self.n_features, 3*self.n_features, 
                       dtype=torch.float64, requires_grad=True) * 0.01
        
        # Enhanced optimizer
        optimizer = torch.optim.AdamW([A, B], lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=100, verbose=True
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        print("Starting training with overfitting prevention...")
        print(f"Parameters: lr={lr}, weight_decay={weight_decay}, λ₁={lambda_l1}, λ₂={lambda_l2}")
        
        for epoch in range(max_epochs):
            # Training
            optimizer.zero_grad()
            
            # Forward pass
            a = A @ self.Y_train[:-1].T
            b = B @ self.X_train[:-1].T
            train_pred = (a + b).T
            train_loss = self.custom_loss(train_pred, self.Y_train[1:], A, B, lambda_l1, lambda_l2)
            
            # Validation
            with torch.no_grad():
                a_val = A @ self.Y_val[:-1].T
                b_val = B @ self.X_val[:-1].T
                val_pred = (a_val + b_val).T
                val_loss = nn.MSELoss()(val_pred, self.Y_val[1:])
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            scheduler.step(val_loss)
            
            # Record losses
            self.train_losses.append(train_loss.item())
            self.val_losses.append(val_loss.item())
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = {
                    'A': A.clone(),
                    'B': B.clone(),
                    'epoch': epoch
                }
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                A.data = self.best_model_state['A'].data
                B.data = self.best_model_state['B'].data
                break
            
            # Progress reporting
            if epoch % 1000 == 0:
                print(f"Epoch {epoch:4d} | Train Loss: {train_loss.item():.6f} | "
                      f"Val Loss: {val_loss.item():.6f} | "
                      f"A.norm: {A.norm():.6f} | B.norm: {B.norm():.6f}")
        
        self.A = A
        self.B = B
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        return self.A, self.B
    
    def evaluate_model(self):
        """Evaluate the model on test data."""
        with torch.no_grad():
            a_test = self.A @ self.Y_test[:-1].T
            b_test = self.B @ self.X_test[:-1].T
            test_pred = (a_test + b_test).T
            test_loss = nn.MSELoss()(test_pred, self.Y_test[1:])
            
            # Calculate R² score
            ss_res = torch.sum((self.Y_test[1:] - test_pred) ** 2)
            ss_tot = torch.sum((self.Y_test[1:] - self.Y_test[1:].mean()) ** 2)
            r2_score = 1 - (ss_res / ss_tot)
            
            print(f"Test Results:")
            print(f"  MSE: {test_loss.item():.6f}")
            print(f"  R²: {r2_score.item():.6f}")
            
            return test_loss.item(), r2_score.item()
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        plt.figure(figsize=(15, 5))
        
        # Full training curves
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Recent training curves
        plt.subplot(1, 3, 2)
        recent_epochs = min(1000, len(self.train_losses))
        plt.plot(self.train_losses[-recent_epochs:], label='Training Loss', alpha=0.7)
        plt.plot(self.val_losses[-recent_epochs:], label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Recent Training Curves')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Loss difference (overfitting indicator)
        plt.subplot(1, 3, 3)
        loss_diff = np.array(self.train_losses) - np.array(self.val_losses)
        plt.plot(loss_diff, label='Train - Val Loss', color='red', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.title('Overfitting Indicator')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, n_samples=100):
        """Plot model predictions vs actual values."""
        with torch.no_grad():
            a_test = self.A @ self.Y_test[:-1].T
            b_test = self.B @ self.X_test[:-1].T
            test_pred = (a_test + b_test).T
            
            # Inverse transform predictions
            test_pred_original = self.scaler_Y.inverse_transform(test_pred.numpy())
            test_actual_original = self.scaler_Y.inverse_transform(self.Y_test[1:].numpy())
            
            plt.figure(figsize=(15, 5))
            
            for i in range(min(2, test_pred_original.shape[1])):
                plt.subplot(1, 2, i+1)
                plt.scatter(test_actual_original[:n_samples, i], 
                           test_pred_original[:n_samples, i], 
                           alpha=0.6, s=20)
                
                # Perfect prediction line
                min_val = min(test_actual_original[:n_samples, i].min(), 
                             test_pred_original[:n_samples, i].min())
                max_val = max(test_actual_original[:n_samples, i].max(), 
                             test_pred_original[:n_samples, i].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title(f'Output {i+1} Predictions')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()


def run_optimized_analysis(X_data, Y_data):
    """
    Run comprehensive optimized analysis.
    
    Parameters:
    -----------
    X_data : np.ndarray
        Input features
    Y_data : np.ndarray
        Target variables
    """
    print("=" * 60)
    print("OPTIMIZED FLUID MODELING ANALYSIS")
    print("=" * 60)
    
    # Initialize modeler
    modeler = OptimizedFluidModeler(X_data, Y_data)
    
    # Train model with overfitting prevention
    A, B = modeler.train_model(
        max_epochs=10000,
        patience=200,
        lr=0.01,
        weight_decay=1e-4,
        lambda_l1=1e-5,
        lambda_l2=1e-4
    )
    
    # Evaluate model
    test_mse, test_r2 = modeler.evaluate_model()
    
    # Plot results
    modeler.plot_training_curves()
    modeler.plot_predictions()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)
    print(f"Final Test MSE: {test_mse:.6f}")
    print(f"Final Test R²: {test_r2:.6f}")
    
    return modeler


if __name__ == "__main__":
    # Example usage
    # Assuming you have X_data and Y_data from your notebook
    # modeler = run_optimized_analysis(X_data, Y_data)
    pass 