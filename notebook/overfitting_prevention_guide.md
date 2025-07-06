# Overfitting Prevention and Advanced Optimization Methods

## Table of Contents
1. [Overfitting Prevention Strategies](#overfitting-prevention-strategies)
2. [Advanced Optimization Methods](#advanced-optimization-methods)
3. [Implementation Examples](#implementation-examples)
4. [Best Practices](#best-practices)
5. [Performance Comparison](#performance-comparison)

## Overfitting Prevention Strategies

### 1. Regularization Techniques

#### L1 Regularization (Lasso)
- **Purpose**: Feature selection and sparse solutions
- **Effect**: Forces some coefficients to exactly zero
- **Use case**: When you suspect many features are irrelevant

```python
# L1 regularization
l1_reg = lambda_l1 * (torch.norm(A, p=1) + torch.norm(B, p=1))
loss = mse_loss + l1_reg
```

#### L2 Regularization (Ridge)
- **Purpose**: Prevents large parameter values
- **Effect**: Shrinks coefficients toward zero
- **Use case**: General overfitting prevention

```python
# L2 regularization (weight decay)
optimizer = torch.optim.AdamW([A, B], lr=0.01, weight_decay=1e-4)
```

### 2. Early Stopping

#### Implementation
```python
patience = 200
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(max_epochs):
    # Training...
    val_loss = compute_validation_loss()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        model.load_state_dict(best_model_state)
        break
```

#### Benefits
- Prevents overfitting by stopping when validation loss stops improving
- Saves computational resources
- Automatically finds optimal training duration

### 3. Data Augmentation and Preprocessing

#### Standardization
```python
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
Y_train_scaled = scaler_Y.fit_transform(Y_train)
```

#### Proper Data Splitting
```python
# Train/Validation/Test split (70/15/15)
train_size = int(0.7 * n_samples)
val_size = int(0.15 * n_samples)

X_train = X[:train_size]
X_val = X[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
```

### 4. Model Complexity Control

#### Parameter Initialization
```python
# Proper initialization prevents vanishing/exploding gradients
A = torch.randn(2*n_features, 2*n_features, dtype=torch.float64, requires_grad=True) * 0.01
B = torch.randn(2*n_features, 3*n_features, dtype=torch.float64, requires_grad=True) * 0.01
```

#### Learning Rate Scheduling
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100, verbose=True
)
```

## Advanced Optimization Methods

### 1. Gradient Descent with Line Search

#### Advantages
- Simple and reliable
- Good for convex problems
- Easy to implement

#### Disadvantages
- Can be slow for large problems
- May get stuck in local minima

#### Implementation
```python
def gradient_descent_with_line_search(lr=0.01, max_iter=10000):
    for i in range(max_iter):
        loss.backward()
        
        # Line search for optimal step size
        best_loss = float('inf')
        best_lr = lr
        
        for step_size in [lr * 0.5, lr, lr * 1.5, lr * 2.0]:
            A_temp = A - step_size * A.grad
            B_temp = B - step_size * B.grad
            loss_temp = compute_loss(A_temp, B_temp)
            
            if loss_temp < best_loss:
                best_loss = loss_temp
                best_lr = step_size
        
        # Update parameters
        A -= best_lr * A.grad
        B -= best_lr * B.grad
```

### 2. Adam Optimizer

#### Advantages
- Adaptive learning rate
- Good for non-convex problems
- Handles sparse gradients well

#### Disadvantages
- More hyperparameters to tune
- Can be sensitive to initialization

#### Implementation
```python
optimizer = torch.optim.Adam([A, B], lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100
)

for epoch in range(max_epochs):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(val_loss)
```

### 3. L-BFGS Optimizer

#### Advantages
- Second-order optimization
- Good convergence properties
- Fewer iterations needed

#### Disadvantages
- Memory intensive
- May not work well with noisy gradients

#### Implementation
```python
optimizer = torch.optim.LBFGS([A, B], lr=0.1, max_iter=20)

def closure():
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()
    return loss

for i in range(max_iter // 20):
    optimizer.step(closure)
```

### 4. Ensemble Methods

#### Advantages
- Robust to local minima
- Better generalization
- Uncertainty quantification

#### Disadvantages
- Computationally expensive
- More complex to implement

#### Implementation
```python
def ensemble_optimization(n_models=5):
    models = []
    
    for model_idx in range(n_models):
        torch.manual_seed(model_idx)  # Different initialization
        A = torch.randn(...) * 0.01
        B = torch.randn(...) * 0.01
        
        # Train model
        train_model(A, B)
        models.append((A, B))
    
    return models
```

### 5. SciPy Optimization

#### Advantages
- Robust optimization algorithms
- Good for constrained problems
- Well-tested implementations

#### Disadvantages
- Requires function conversion
- May be slower for large problems

#### Implementation
```python
from scipy.optimize import minimize

def objective(params):
    # Reshape parameters
    A_flat = params[:n_A].reshape(...)
    B_flat = params[n_A:].reshape(...)
    
    # Compute predictions and loss
    pred = A_flat @ Y + B_flat @ X
    return np.mean((pred - target) ** 2)

result = minimize(objective, initial_params, method='L-BFGS-B')
```

## Implementation Examples

### Complete Overfitting Prevention Pipeline

```python
class OptimizedFluidModeler:
    def __init__(self, X_data, Y_data, train_ratio=0.7, val_ratio=0.15):
        self.X_data = X_data
        self.Y_data = Y_data
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Data preprocessing
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        # Split data
        self._prepare_data()
    
    def custom_loss(self, pred, target, A, B, lambda_l1=1e-5, lambda_l2=1e-4):
        mse_loss = nn.MSELoss()(pred, target)
        l1_reg = lambda_l1 * (torch.norm(A, p=1) + torch.norm(B, p=1))
        l2_reg = lambda_l2 * (torch.norm(A, p=2)**2 + torch.norm(B, p=2)**2)
        return mse_loss + l1_reg + l2_reg
    
    def train_model(self, max_epochs=10000, patience=200, lr=0.01):
        # Initialize parameters
        A = torch.randn(...) * 0.01
        B = torch.randn(...) * 0.01
        
        # Enhanced optimizer
        optimizer = torch.optim.AdamW([A, B], lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=100
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training
            train_loss = self.custom_loss(train_pred, train_target, A, B)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            val_loss = compute_validation_loss()
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {'A': A.clone(), 'B': B.clone()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                A.data = best_model_state['A'].data
                B.data = best_model_state['B'].data
                break
```

## Best Practices

### 1. Hyperparameter Tuning

#### Learning Rate
- Start with `lr=0.01`
- Use learning rate scheduling
- Monitor training curves

#### Regularization
- L1: `lambda_l1=1e-5` to `1e-3`
- L2: `lambda_l2=1e-4` to `1e-2`
- Weight decay: `1e-4` to `1e-3`

#### Early Stopping
- Patience: 100-200 epochs
- Monitor validation loss
- Save best model state

### 2. Data Preprocessing

#### Scaling
- Always standardize/normalize data
- Use separate scalers for train/val/test
- Apply same transformation to new data

#### Splitting
- Use stratified split if possible
- Maintain temporal order for time series
- Keep validation set separate from test set

### 3. Monitoring and Debugging

#### Training Curves
```python
plt.figure(figsize=(15, 5))

# Training and validation loss
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.yscale('log')

# Loss difference (overfitting indicator)
plt.subplot(1, 3, 2)
loss_diff = np.array(train_losses) - np.array(val_losses)
plt.plot(loss_diff, label='Train - Val Loss', color='red')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss Difference')
plt.title('Overfitting Indicator')
plt.legend()

# Parameter norms
plt.subplot(1, 3, 3)
plt.plot(A_norms, label='A norm')
plt.plot(B_norms, label='B norm')
plt.xlabel('Epoch')
plt.ylabel('Parameter Norm')
plt.title('Parameter Stability')
plt.legend()

plt.tight_layout()
plt.show()
```

### 4. Model Selection

#### Cross-Validation
- Use k-fold cross-validation for small datasets
- Use time series cross-validation for temporal data
- Compare multiple optimization methods

#### Ensemble Methods
- Train multiple models with different initializations
- Use averaging or voting for predictions
- Quantify uncertainty

## Performance Comparison

### Method Comparison Table

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Gradient Descent | Simple, reliable | Slow, local minima | Small problems |
| Adam | Fast, adaptive | Hyperparameter sensitive | Most problems |
| L-BFGS | Few iterations, good convergence | Memory intensive | Smooth objectives |
| Regularized | Prevents overfitting | More hyperparameters | High-dimensional data |
| Ensemble | Robust, better generalization | Expensive | Critical applications |
| SciPy | Well-tested, robust | Function conversion | Constrained problems |

### Recommended Workflow

1. **Start with Adam optimizer** - Good default choice
2. **Add regularization** - L2 weight decay first, then L1 if needed
3. **Implement early stopping** - Monitor validation loss
4. **Try ensemble methods** - If computational resources allow
5. **Compare methods** - Use cross-validation for fair comparison

### Code Usage

```python
# Quick start with overfitting prevention
from optimized_ml_modelling import OptimizedFluidModeler

# Initialize modeler
modeler = OptimizedFluidModeler(X_data, Y_data)

# Train with comprehensive overfitting prevention
A, B = modeler.train_model(
    max_epochs=10000,
    patience=200,
    lr=0.01,
    weight_decay=1e-4,
    lambda_l1=1e-5,
    lambda_l2=1e-4
)

# Evaluate and plot results
test_mse, test_r2 = modeler.evaluate_model()
modeler.plot_training_curves()
modeler.plot_predictions()
```

This comprehensive approach ensures robust model training with minimal overfitting and optimal performance. 