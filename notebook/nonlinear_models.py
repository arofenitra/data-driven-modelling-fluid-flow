#!/usr/bin/env python3
"""
Nonlinear Models for Fluid Flow Modeling
=======================================

This module provides various nonlinear modeling approaches that can
significantly outperform linear regression for complex fluid dynamics.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class NonlinearFluidModeler:
    """
    Comprehensive nonlinear modeling for fluid flow dynamics.
    """
    
    def __init__(self, X_data, Y_data, t_data=None, x_data=None):
        """
        Initialize the nonlinear fluid modeler.
        
        Parameters:
        -----------
        X_data : np.ndarray
            Input features (pressure, water holdup, gas holdup)
        Y_data : np.ndarray
            Target variables (water velocity, gas velocity)
        t_data : np.ndarray, optional
            Time coordinates
        x_data : np.ndarray, optional
            Spatial coordinates
        """
        self.X_data = X_data
        self.Y_data = Y_data
        self.t_data = t_data
        self.x_data = x_data
        
        # Data preprocessing
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        # Model storage
        self.models = {}
        self.results = {}
        
    def prepare_data(self, T_train=None, d=1):
        """Prepare data for modeling."""
        if T_train is None:
            T_train = 3 * self.X_data.shape[1] // 4
            
        self.T_train = T_train
        self.d = d
        
        # Prepare training data
        Y_train = self.Y_data[:, d:T_train]
        Z_train = np.vstack([self.X_data[:, i:T_train-d+i] for i in range(d)])
        
        # Prepare test data
        Z_test = np.vstack([self.X_data[:, T_train+i:self.X_data.shape[1]-d+i+1] for i in range(d)])
        Y_test = self.Y_data[:, T_train:]
        
        # Scale data
        Z_train_scaled = self.scaler_X.fit_transform(Z_train.T)
        Y_train_scaled = self.scaler_Y.fit_transform(Y_train.T)
        Z_test_scaled = self.scaler_X.transform(Z_test.T)
        
        self.Z_train = Z_train_scaled
        self.Y_train = Y_train_scaled
        self.Z_test = Z_test_scaled
        self.Y_test = Y_test
        
        print(f"Data shapes:")
        print(f"  Z_train: {self.Z_train.shape}")
        print(f"  Y_train: {self.Y_train.shape}")
        print(f"  Z_test: {self.Z_test.shape}")
        print(f"  Y_test: {self.Y_test.shape}")
        
        return self.Z_train, self.Y_train, self.Z_test, self.Y_test
    
    def model_1_polynomial_regression(self, degree=3):
        """
        Model 1: Polynomial Regression
        
        Advantages:
        - Captures nonlinear relationships
        - Interpretable coefficients
        - No hyperparameter tuning needed
        
        Disadvantages:
        - Can overfit with high degrees
        - Limited to polynomial relationships
        """
        print("Model 1: Polynomial Regression")
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Z_train_poly = poly.fit_transform(self.Z_train)
        Z_test_poly = poly.transform(self.Z_test)
        
        # Fit polynomial regression
        A_poly = np.linalg.pinv(Z_train_poly) @ self.Y_train
        
        # Predictions
        Y_train_pred = Z_train_poly @ A_poly
        Y_test_pred = Z_test_poly @ A_poly
        
        # Transform back to original scale
        Y_train_pred_orig = self.scaler_Y.inverse_transform(Y_train_pred)
        Y_test_pred_orig = self.scaler_Y.inverse_transform(Y_test_pred)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.Y_train, Y_train_pred)
        test_mse = mean_squared_error(self.Y_test, Y_test_pred_orig)
        train_r2 = r2_score(self.Y_train, Y_train_pred)
        test_r2 = r2_score(self.Y_test, Y_test_pred_orig)
        
        self.models['polynomial'] = {
            'poly': poly,
            'coefficients': A_poly,
            'degree': degree
        }
        
        self.results['polynomial'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': Y_test_pred_orig
        }
        
        print(f"  Train MSE: {train_mse:.6f}")
        print(f"  Test MSE: {test_mse:.6f}")
        print(f"  Train R²: {train_r2:.6f}")
        print(f"  Test R²: {test_r2:.6f}")
        
        return Y_test_pred_orig
    
    def model_2_neural_network(self, hidden_layers=(100, 50), activation='relu', 
                              max_iter=1000, learning_rate=0.001):
        """
        Model 2: Neural Network
        
        Advantages:
        - Universal function approximator
        - Can capture complex nonlinearities
        - Good for high-dimensional data
        
        Disadvantages:
        - Black box model
        - Requires hyperparameter tuning
        - Can overfit easily
        """
        print("Model 2: Neural Network")
        
        # Create neural network
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            max_iter=max_iter,
            learning_rate_init=learning_rate,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Fit model
        model.fit(self.Z_train, self.Y_train)
        
        # Predictions
        Y_train_pred = model.predict(self.Z_train)
        Y_test_pred = model.predict(self.Z_test)
        
        # Transform back to original scale
        Y_train_pred_orig = self.scaler_Y.inverse_transform(Y_train_pred)
        Y_test_pred_orig = self.scaler_Y.inverse_transform(Y_test_pred)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.Y_train, Y_train_pred)
        test_mse = mean_squared_error(self.Y_test, Y_test_pred_orig)
        train_r2 = r2_score(self.Y_train, Y_train_pred)
        test_r2 = r2_score(self.Y_test, Y_test_pred_orig)
        
        self.models['neural_network'] = model
        self.results['neural_network'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': Y_test_pred_orig
        }
        
        print(f"  Train MSE: {train_mse:.6f}")
        print(f"  Test MSE: {test_mse:.6f}")
        print(f"  Train R²: {train_r2:.6f}")
        print(f"  Test R²: {test_r2:.6f}")
        
        return Y_test_pred_orig
    
    def model_3_random_forest(self, n_estimators=100, max_depth=None):
        """
        Model 3: Random Forest
        
        Advantages:
        - Handles nonlinear relationships
        - Feature importance ranking
        - Robust to outliers
        - No feature scaling needed
        
        Disadvantages:
        - Less interpretable than linear models
        - Can be computationally expensive
        """
        print("Model 3: Random Forest")
        
        # Train separate models for each output
        models = []
        predictions = []
        
        for i in range(self.Y_train.shape[1]):
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(self.Z_train, self.Y_train[:, i])
            models.append(rf)
            predictions.append(rf.predict(self.Z_test))
        
        # Combine predictions
        Y_test_pred = np.column_stack(predictions)
        Y_train_pred = np.column_stack([model.predict(self.Z_train) for model in models])
        
        # Transform back to original scale
        Y_test_pred_orig = self.scaler_Y.inverse_transform(Y_test_pred)
        Y_train_pred_orig = self.scaler_Y.inverse_transform(Y_train_pred)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.Y_train, Y_train_pred)
        test_mse = mean_squared_error(self.Y_test, Y_test_pred_orig)
        train_r2 = r2_score(self.Y_train, Y_train_pred)
        test_r2 = r2_score(self.Y_test, Y_test_pred_orig)
        
        self.models['random_forest'] = models
        self.results['random_forest'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': Y_test_pred_orig,
            'feature_importance': [model.feature_importances_ for model in models]
        }
        
        print(f"  Train MSE: {train_mse:.6f}")
        print(f"  Test MSE: {test_mse:.6f}")
        print(f"  Train R²: {train_r2:.6f}")
        print(f"  Test R²: {test_r2:.6f}")
        
        return Y_test_pred_orig
    
    def model_4_deep_neural_network(self, hidden_layers=(128, 64, 32), 
                                   dropout_rate=0.2, epochs=1000, batch_size=32):
        """
        Model 4: Deep Neural Network with PyTorch
        
        Advantages:
        - Very flexible architecture
        - Can capture complex patterns
        - Good for large datasets
        
        Disadvantages:
        - Requires significant tuning
        - Computationally expensive
        - Risk of overfitting
        """
        print("Model 4: Deep Neural Network")
        
        # Convert to PyTorch tensors
        X_train_torch = torch.tensor(self.Z_train, dtype=torch.float32)
        Y_train_torch = torch.tensor(self.Y_train, dtype=torch.float32)
        X_test_torch = torch.tensor(self.Z_test, dtype=torch.float32)
        
        # Define neural network
        class DeepNN(nn.Module):
            def __init__(self, input_size, hidden_layers, output_size, dropout_rate):
                super(DeepNN, self).__init__()
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_layers:
                    layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.BatchNorm1d(hidden_size)
                    ])
                    prev_size = hidden_size
                
                layers.append(nn.Linear(prev_size, output_size))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        # Initialize model
        model = DeepNN(
            input_size=self.Z_train.shape[1],
            hidden_layers=hidden_layers,
            output_size=self.Y_train.shape[1],
            dropout_rate=dropout_rate
        )
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_train_torch)
            loss = criterion(outputs, Y_train_torch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                model.load_state_dict(best_model_state)
                break
            
            if epoch % 100 == 0:
                print(f"    Epoch {epoch}: Loss = {loss.item():.6f}")
        
        # Predictions
        model.eval()
        with torch.no_grad():
            Y_train_pred = model(X_train_torch).numpy()
            Y_test_pred = model(X_test_torch).numpy()
        
        # Transform back to original scale
        Y_train_pred_orig = self.scaler_Y.inverse_transform(Y_train_pred)
        Y_test_pred_orig = self.scaler_Y.inverse_transform(Y_test_pred)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.Y_train, Y_train_pred)
        test_mse = mean_squared_error(self.Y_test, Y_test_pred_orig)
        train_r2 = r2_score(self.Y_train, Y_train_pred)
        test_r2 = r2_score(self.Y_test, Y_test_pred_orig)
        
        self.models['deep_nn'] = model
        self.results['deep_nn'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': Y_test_pred_orig
        }
        
        print(f"  Train MSE: {train_mse:.6f}")
        print(f"  Test MSE: {test_mse:.6f}")
        print(f"  Train R²: {train_r2:.6f}")
        print(f"  Test R²: {test_r2:.6f}")
        
        return Y_test_pred_orig
    
    def model_5_ensemble(self, models_to_use=['polynomial', 'neural_network', 'random_forest']):
        """
        Model 5: Ensemble of Multiple Models
        
        Advantages:
        - Combines strengths of different models
        - More robust predictions
        - Reduces overfitting
        
        Disadvantages:
        - More complex
        - Requires more computational resources
        """
        print("Model 5: Ensemble Model")
        
        # Train individual models
        predictions = []
        model_names = []
        
        for model_name in models_to_use:
            if model_name == 'polynomial':
                pred = self.model_1_polynomial_regression()
                predictions.append(pred)
                model_names.append('Polynomial')
            elif model_name == 'neural_network':
                pred = self.model_2_neural_network()
                predictions.append(pred)
                model_names.append('Neural Network')
            elif model_name == 'random_forest':
                pred = self.model_3_random_forest()
                predictions.append(pred)
                model_names.append('Random Forest')
        
        # Ensemble prediction (simple averaging)
        Y_test_pred_ensemble = np.mean(predictions, axis=0)
        
        # Calculate metrics
        test_mse = mean_squared_error(self.Y_test, Y_test_pred_ensemble)
        test_r2 = r2_score(self.Y_test, Y_test_pred_ensemble)
        
        self.results['ensemble'] = {
            'test_mse': test_mse,
            'test_r2': test_r2,
            'predictions': Y_test_pred_ensemble,
            'individual_predictions': predictions,
            'model_names': model_names
        }
        
        print(f"  Test MSE: {test_mse:.6f}")
        print(f"  Test R²: {test_r2:.6f}")
        
        return Y_test_pred_ensemble
    
    def compare_models(self):
        """Compare all trained models."""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            if 'test_mse' in results:
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Test MSE': results['test_mse'],
                    'Test R²': results['test_r2']
                })
        
        # Sort by test MSE
        comparison_data.sort(key=lambda x: x['Test MSE'])
        
        print(f"{'Model':<20} {'Test MSE':<12} {'Test R²':<8}")
        print("-" * 40)
        
        for data in comparison_data:
            print(f"{data['Model']:<20} {data['Test MSE']:<12.6f} {data['Test R²']:<8.4f}")
        
        return comparison_data
    
    def plot_comparison(self):
        """Plot comparison of model predictions."""
        n_models = len(self.results)
        if n_models == 0:
            print("No models to compare. Train models first.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Test MSE comparison
        model_names = []
        test_mses = []
        
        for model_name, results in self.results.items():
            if 'test_mse' in results:
                model_names.append(model_name.replace('_', ' ').title())
                test_mses.append(results['test_mse'])
        
        axes[0].bar(model_names, test_mses)
        axes[0].set_title('Test MSE Comparison')
        axes[0].set_ylabel('MSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Test R² comparison
        test_r2s = []
        for model_name, results in self.results.items():
            if 'test_r2' in results:
                test_r2s.append(results['test_r2'])
        
        axes[1].bar(model_names, test_r2s)
        axes[1].set_title('Test R² Comparison')
        axes[1].set_ylabel('R²')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Predictions vs Actual (best model)
        best_model = min(self.results.items(), key=lambda x: x[1].get('test_mse', float('inf')))
        if 'predictions' in best_model[1]:
            predictions = best_model[1]['predictions']
            actual = self.Y_test
            
            axes[2].scatter(actual.flatten(), predictions.flatten(), alpha=0.6, s=20)
            min_val = min(actual.min(), predictions.min())
            max_val = max(actual.max(), predictions.max())
            axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            axes[2].set_xlabel('Actual Values')
            axes[2].set_ylabel('Predicted Values')
            axes[2].set_title(f'Best Model: {best_model[0].replace("_", " ").title()}')
        
        # Plot 4: Time series comparison (if available)
        if self.t_data is not None and 'predictions' in best_model[1]:
            t_test = self.t_data[self.T_train:]
            axes[3].plot(t_test, actual[0, :], label='Actual', alpha=0.7)
            axes[3].plot(t_test, predictions[0, :], '--', label='Predicted', alpha=0.7)
            axes[3].set_xlabel('Time')
            axes[3].set_ylabel('Value')
            axes[3].set_title('Time Series Comparison')
            axes[3].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model_name='random_forest'):
        """Plot feature importance for models that support it."""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return
        
        if model_name == 'random_forest':
            models = self.models[model_name]
            feature_importance = self.results[model_name]['feature_importance']
            
            plt.figure(figsize=(12, 6))
            
            for i, importance in enumerate(feature_importance):
                plt.subplot(1, len(feature_importance), i+1)
                plt.bar(range(len(importance)), importance)
                plt.title(f'Output {i+1} Feature Importance')
                plt.xlabel('Feature Index')
                plt.ylabel('Importance')
            
            plt.tight_layout()
            plt.show()


def run_nonlinear_analysis(X_data, Y_data, t_data=None, x_data=None):
    """
    Run comprehensive nonlinear analysis.
    
    Parameters:
    -----------
    X_data : np.ndarray
        Input features
    Y_data : np.ndarray
        Target variables
    t_data : np.ndarray, optional
        Time coordinates
    x_data : np.ndarray, optional
        Spatial coordinates
    """
    print("=" * 60)
    print("NONLINEAR FLUID MODELING ANALYSIS")
    print("=" * 60)
    
    # Initialize modeler
    modeler = NonlinearFluidModeler(X_data, Y_data, t_data, x_data)
    
    # Prepare data
    modeler.prepare_data()
    
    # Train different models
    print("\nTraining models...")
    
    # Polynomial regression
    modeler.model_1_polynomial_regression(degree=3)
    
    # Neural network
    modeler.model_2_neural_network(hidden_layers=(50, 25))
    
    # Random forest
    modeler.model_3_random_forest(n_estimators=100)
    
    # Deep neural network
    modeler.model_4_deep_neural_network(hidden_layers=(64, 32), epochs=500)
    
    # Ensemble
    modeler.model_5_ensemble()
    
    # Compare models
    comparison = modeler.compare_models()
    
    # Plot results
    modeler.plot_comparison()
    
    # Plot feature importance
    modeler.plot_feature_importance()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)
    
    return modeler, comparison


if __name__ == "__main__":
    # Example usage
    # modeler, comparison = run_nonlinear_analysis(X_data, Y_data, t_data)
    pass 