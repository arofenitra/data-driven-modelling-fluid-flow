#!/usr/bin/env python3
"""
Advanced Optimization Methods for Fluid Modeling
===============================================

This module provides various advanced optimization techniques and methods
to improve the performance of fluid flow modeling.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedFluidOptimizer:
    """
    Advanced optimization methods for fluid modeling with multiple algorithms.
    """
    
    def __init__(self, X_data, Y_data):
        """
        Initialize the advanced fluid optimizer.
        
        Parameters:
        -----------
        X_data : np.ndarray
            Input features
        Y_data : np.ndarray
            Target variables
        """
        self.X_data = X_data
        self.Y_data = Y_data
        self.n_features = X_data.shape[0] // 3
        self.n_outputs = Y_data.shape[0] // 2
        
        # Data preprocessing
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for optimization."""
        X_reshaped = self.X_data.T
        Y_reshaped = self.Y_data.T
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X_reshaped)
        Y_scaled = self.scaler_Y.fit_transform(Y_reshaped)
        
        # Convert to tensors
        self.X_tensor = torch.tensor(X_scaled, dtype=torch.float64)
        self.Y_tensor = torch.tensor(Y_scaled, dtype=torch.float64)
        
    def method_1_gradient_descent(self, lr=0.01, max_iter=10000, tol=1e-6):
        """
        Method 1: Gradient Descent with Line Search
        
        Advantages:
        - Simple and reliable
        - Good for convex problems
        - Easy to implement
        
        Disadvantages:
        - Can be slow for large problems
        - May get stuck in local minima
        """
        print("Method 1: Gradient Descent with Line Search")
        
        # Initialize parameters
        A = torch.randn(2*self.n_features, 2*self.n_features, 
                       dtype=torch.float64, requires_grad=True) * 0.01
        B = torch.randn(2*self.n_features, 3*self.n_features, 
                       dtype=torch.float64, requires_grad=True) * 0.01
        
        losses = []
        
        for i in range(max_iter):
            # Forward pass
            pred = A @ self.Y_tensor[:-1].T + B @ self.X_tensor[:-1].T
            loss = nn.MSELoss()(pred.T, self.Y_tensor[1:])
            
            # Backward pass
            loss.backward()
            
            # Gradient descent with line search
            with torch.no_grad():
                # Store gradients
                A_grad = A.grad.clone()
                B_grad = B.grad.clone()
                
                # Line search for optimal step size
                best_loss = float('inf')
                best_lr = lr
                
                for step_size in [lr * 0.5, lr, lr * 1.5, lr * 2.0]:
                    A_temp = A - step_size * A_grad
                    B_temp = B - step_size * B_grad
                    
                    pred_temp = A_temp @ self.Y_tensor[:-1].T + B_temp @ self.X_tensor[:-1].T
                    loss_temp = nn.MSELoss()(pred_temp.T, self.Y_tensor[1:])
                    
                    if loss_temp < best_loss:
                        best_loss = loss_temp
                        best_lr = step_size
                
                # Update parameters
                A -= best_lr * A_grad
                B -= best_lr * B_grad
                
                # Zero gradients
                A.grad.zero_()
                B.grad.zero_()
            
            losses.append(loss.item())
            
            if i % 1000 == 0:
                print(f"Iteration {i}: Loss = {loss.item():.6f}, LR = {best_lr:.6f}")
            
            # Convergence check
            if i > 0 and abs(losses[-1] - losses[-2]) < tol:
                print(f"Converged at iteration {i}")
                break
        
        return A, B, losses
    
    def method_2_adam_optimizer(self, lr=0.01, max_iter=10000, weight_decay=1e-4):
        """
        Method 2: Adam Optimizer with Adaptive Learning Rate
        
        Advantages:
        - Adaptive learning rate
        - Good for non-convex problems
        - Handles sparse gradients well
        
        Disadvantages:
        - More hyperparameters to tune
        - Can be sensitive to initialization
        """
        print("Method 2: Adam Optimizer with Adaptive Learning Rate")
        
        # Initialize parameters
        A = torch.randn(2*self.n_features, 2*self.n_features, 
                       dtype=torch.float64, requires_grad=True) * 0.01
        B = torch.randn(2*self.n_features, 3*self.n_features, 
                       dtype=torch.float64, requires_grad=True) * 0.01
        
        # Adam optimizer
        optimizer = torch.optim.Adam([A, B], lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=100, verbose=True
        )
        
        losses = []
        
        for i in range(max_iter):
            optimizer.zero_grad()
            
            # Forward pass
            pred = A @ self.Y_tensor[:-1].T + B @ self.X_tensor[:-1].T
            loss = nn.MSELoss()(pred.T, self.Y_tensor[1:])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            losses.append(loss.item())
            
            if i % 1000 == 0:
                print(f"Iteration {i}: Loss = {loss.item():.6f}")
        
        return A, B, losses
    
    def method_3_lbfgs_optimizer(self, max_iter=10000):
        """
        Method 3: L-BFGS Optimizer (Limited-memory BFGS)
        
        Advantages:
        - Second-order optimization
        - Good convergence properties
        - Fewer iterations needed
        
        Disadvantages:
        - Memory intensive
        - May not work well with noisy gradients
        """
        print("Method 3: L-BFGS Optimizer")
        
        # Initialize parameters
        A = torch.randn(2*self.n_features, 2*self.n_features, 
                       dtype=torch.float64, requires_grad=True) * 0.01
        B = torch.randn(2*self.n_features, 3*self.n_features, 
                       dtype=torch.float64, requires_grad=True) * 0.01
        
        # L-BFGS optimizer
        optimizer = torch.optim.LBFGS([A, B], lr=0.1, max_iter=20)
        
        losses = []
        
        def closure():
            optimizer.zero_grad()
            pred = A @ self.Y_tensor[:-1].T + B @ self.X_tensor[:-1].T
            loss = nn.MSELoss()(pred.T, self.Y_tensor[1:])
            loss.backward()
            losses.append(loss.item())
            return loss
        
        for i in range(max_iter // 20):  # L-BFGS uses closure function
            optimizer.step(closure)
            
            if i % 50 == 0:
                print(f"Iteration {i*20}: Loss = {losses[-1]:.6f}")
        
        return A, B, losses
    
    def method_4_regularized_optimization(self, lambda_l1=1e-5, lambda_l2=1e-4, 
                                        max_iter=10000, lr=0.01):
        """
        Method 4: Regularized Optimization with L1/L2 Penalties
        
        Advantages:
        - Prevents overfitting
        - Feature selection (L1)
        - Better generalization
        
        Disadvantages:
        - More hyperparameters
        - May underfit if regularization is too strong
        """
        print("Method 4: Regularized Optimization with L1/L2 Penalties")
        
        # Initialize parameters
        A = torch.randn(2*self.n_features, 2*self.n_features, 
                       dtype=torch.float64, requires_grad=True) * 0.01
        B = torch.randn(2*self.n_features, 3*self.n_features, 
                       dtype=torch.float64, requires_grad=True) * 0.01
        
        optimizer = torch.optim.AdamW([A, B], lr=lr, weight_decay=lambda_l2)
        
        losses = []
        
        for i in range(max_iter):
            optimizer.zero_grad()
            
            # Forward pass
            pred = A @ self.Y_tensor[:-1].T + B @ self.X_tensor[:-1].T
            mse_loss = nn.MSELoss()(pred.T, self.Y_tensor[1:])
            
            # L1 regularization
            l1_reg = lambda_l1 * (torch.norm(A, p=1) + torch.norm(B, p=1))
            
            # Total loss
            loss = mse_loss + l1_reg
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if i % 1000 == 0:
                print(f"Iteration {i}: MSE = {mse_loss.item():.6f}, L1 = {l1_reg.item():.6f}")
        
        return A, B, losses
    
    def method_5_ensemble_optimization(self, n_models=5, max_iter=5000):
        """
        Method 5: Ensemble Optimization
        
        Advantages:
        - Robust to local minima
        - Better generalization
        - Uncertainty quantification
        
        Disadvantages:
        - Computationally expensive
        - More complex to implement
        """
        print("Method 5: Ensemble Optimization")
        
        models = []
        losses_list = []
        
        for model_idx in range(n_models):
            print(f"Training ensemble model {model_idx + 1}/{n_models}")
            
            # Initialize parameters with different seeds
            torch.manual_seed(model_idx)
            A = torch.randn(2*self.n_features, 2*self.n_features, 
                           dtype=torch.float64, requires_grad=True) * 0.01
            B = torch.randn(2*self.n_features, 3*self.n_features, 
                           dtype=torch.float64, requires_grad=True) * 0.01
            
            optimizer = torch.optim.Adam([A, B], lr=0.01)
            losses = []
            
            for i in range(max_iter):
                optimizer.zero_grad()
                
                pred = A @ self.Y_tensor[:-1].T + B @ self.X_tensor[:-1].T
                loss = nn.MSELoss()(pred.T, self.Y_tensor[1:])
                
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                if i % 1000 == 0:
                    print(f"  Model {model_idx + 1}, Iteration {i}: Loss = {loss.item():.6f}")
            
            models.append((A, B))
            losses_list.append(losses)
        
        return models, losses_list
    
    def method_6_scipy_optimization(self):
        """
        Method 6: SciPy Optimization (L-BFGS-B)
        
        Advantages:
        - Robust optimization algorithms
        - Good for constrained problems
        - Well-tested implementations
        
        Disadvantages:
        - Requires function conversion
        - May be slower for large problems
        """
        print("Method 6: SciPy Optimization (L-BFGS-B)")
        
        # Convert to numpy for scipy
        X_np = self.X_tensor.numpy()
        Y_np = self.Y_tensor.numpy()
        
        # Define objective function
        def objective(params):
            # Reshape parameters
            n_A = 2*self.n_features * 2*self.n_features
            A_flat = params[:n_A].reshape(2*self.n_features, 2*self.n_features)
            B_flat = params[n_A:].reshape(2*self.n_features, 3*self.n_features)
            
            # Compute predictions
            pred = A_flat @ Y_np[:-1].T + B_flat @ X_np[:-1].T
            
            # Compute MSE
            mse = np.mean((pred.T - Y_np[1:]) ** 2)
            return mse
        
        # Initial parameters
        n_A = 2*self.n_features * 2*self.n_features
        n_B = 2*self.n_features * 3*self.n_features
        initial_params = np.random.randn(n_A + n_B) * 0.01
        
        # Optimize
        result = minimize(objective, initial_params, method='L-BFGS-B', 
                        options={'maxiter': 1000, 'disp': True})
        
        # Reshape results
        A_opt = result.x[:n_A].reshape(2*self.n_features, 2*self.n_features)
        B_opt = result.x[n_A:].reshape(2*self.n_features, 3*self.n_features)
        
        return torch.tensor(A_opt, dtype=torch.float64), torch.tensor(B_opt, dtype=torch.float64), [result.fun]
    
    def compare_methods(self):
        """Compare all optimization methods."""
        print("=" * 60)
        print("COMPARING OPTIMIZATION METHODS")
        print("=" * 60)
        
        methods = [
            ("Gradient Descent", self.method_1_gradient_descent),
            ("Adam Optimizer", self.method_2_adam_optimizer),
            ("L-BFGS", self.method_3_lbfgs_optimizer),
            ("Regularized", self.method_4_regularized_optimization),
            ("Ensemble", self.method_5_ensemble_optimization),
            ("SciPy L-BFGS-B", self.method_6_scipy_optimization)
        ]
        
        results = {}
        
        for name, method in methods:
            print(f"\nTesting {name}...")
            try:
                if name == "Ensemble":
                    models, losses_list = method()
                    # Use average of final losses
                    final_loss = np.mean([losses[-1] for losses in losses_list])
                    results[name] = {
                        'final_loss': final_loss,
                        'models': models,
                        'losses': losses_list
                    }
                elif name == "SciPy L-BFGS-B":
                    A, B, losses = method()
                    results[name] = {
                        'final_loss': losses[0],
                        'A': A,
                        'B': B,
                        'losses': losses
                    }
                else:
                    A, B, losses = method()
                    results[name] = {
                        'final_loss': losses[-1],
                        'A': A,
                        'B': B,
                        'losses': losses
                    }
                print(f"  Final Loss: {results[name]['final_loss']:.6f}")
            except Exception as e:
                print(f"  Error: {e}")
                results[name] = {'final_loss': float('inf')}
        
        # Print comparison
        print("\n" + "=" * 60)
        print("METHOD COMPARISON")
        print("=" * 60)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['final_loss'])
        
        for i, (name, result) in enumerate(sorted_results):
            if result['final_loss'] != float('inf'):
                print(f"{i+1}. {name:20s}: {result['final_loss']:.6f}")
        
        return results
    
    def plot_comparison(self, results):
        """Plot comparison of optimization methods."""
        plt.figure(figsize=(15, 10))
        
        # Plot training curves
        plt.subplot(2, 2, 1)
        for name, result in results.items():
            if 'losses' in result and len(result['losses']) > 1:
                if isinstance(result['losses'][0], list):  # Ensemble
                    for i, losses in enumerate(result['losses']):
                        plt.plot(losses, alpha=0.3, label=f'{name} (Model {i+1})' if i == 0 else None)
                else:
                    plt.plot(result['losses'], label=name, alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Curves Comparison')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot final losses
        plt.subplot(2, 2, 2)
        names = []
        losses = []
        for name, result in results.items():
            if result['final_loss'] != float('inf'):
                names.append(name)
                losses.append(result['final_loss'])
        
        bars = plt.bar(range(len(names)), losses)
        plt.xlabel('Method')
        plt.ylabel('Final Loss')
        plt.title('Final Loss Comparison')
        plt.xticks(range(len(names)), names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Color bars by performance
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Plot convergence speed
        plt.subplot(2, 2, 3)
        for name, result in results.items():
            if 'losses' in result and len(result['losses']) > 1:
                if isinstance(result['losses'][0], list):  # Ensemble
                    avg_losses = np.mean(result['losses'], axis=0)
                    plt.plot(avg_losses, label=name, alpha=0.7)
                else:
                    plt.plot(result['losses'], label=name, alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Convergence Speed')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot method characteristics
        plt.subplot(2, 2, 4)
        characteristics = {
            'Gradient Descent': 'Simple, Reliable',
            'Adam Optimizer': 'Adaptive, Fast',
            'L-BFGS': 'Second-order, Few iterations',
            'Regularized': 'Prevents overfitting',
            'Ensemble': 'Robust, Multiple models',
            'SciPy L-BFGS-B': 'Robust, Well-tested'
        }
        
        y_pos = np.arange(len(characteristics))
        plt.barh(y_pos, [1]*len(characteristics))
        plt.yticks(y_pos, list(characteristics.keys()))
        plt.xlabel('Method Type')
        plt.title('Method Characteristics')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def run_advanced_optimization_analysis(X_data, Y_data):
    """
    Run comprehensive advanced optimization analysis.
    
    Parameters:
    -----------
    X_data : np.ndarray
        Input features
    Y_data : np.ndarray
        Target variables
    """
    print("=" * 60)
    print("ADVANCED OPTIMIZATION ANALYSIS")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = AdvancedFluidOptimizer(X_data, Y_data)
    
    # Compare methods
    results = optimizer.compare_methods()
    
    # Plot comparison
    optimizer.plot_comparison(results)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)
    
    return optimizer, results


if __name__ == "__main__":
    # Example usage
    # Assuming you have X_data and Y_data from your notebook
    # optimizer, results = run_advanced_optimization_analysis(X_data, Y_data)
    pass 