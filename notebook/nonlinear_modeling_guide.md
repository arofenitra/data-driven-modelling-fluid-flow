# Nonlinear Modeling for Fluid Flow Dynamics

## Table of Contents
1. [Why Nonlinear Models?](#why-nonlinear-models)
2. [Types of Nonlinear Models](#types-of-nonlinear-models)
3. [Implementation Examples](#implementation-examples)
4. [Model Comparison](#model-comparison)
5. [Best Practices](#best-practices)

## Why Nonlinear Models?

### Limitations of Linear Regression

Your current linear model assumes:
$$y_t = A \cdot x_t$$

**Problems with this approach:**
1. **Oversimplified**: Fluid dynamics are inherently nonlinear
2. **Limited expressiveness**: Cannot capture complex interactions
3. **Poor generalization**: May not work well on unseen data
4. **Missing physics**: Real fluid flows have nonlinear phenomena

### Benefits of Nonlinear Models

**Advantages:**
- **Captures complex relationships**: Can model nonlinear interactions
- **Better accuracy**: Often significantly outperforms linear models
- **Physics-aware**: Can incorporate domain knowledge
- **Robust**: Better generalization to new conditions

## Types of Nonlinear Models

### 1. Polynomial Regression

**Model Form:**
$$y_t = \sum_{i=1}^{d} \sum_{j=1}^{p} \beta_{ij} x_i^j + \text{interaction terms}$$

**Advantages:**
- Simple and interpretable
- Captures polynomial relationships
- No hyperparameter tuning needed

**Disadvantages:**
- Can overfit with high degrees
- Limited to polynomial relationships
- Curse of dimensionality

**Implementation:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X_train)

# Fit model
model = LinearRegression()
model.fit(X_poly, y_train)

# Predictions
X_test_poly = poly.transform(X_test)
predictions = model.predict(X_test_poly)
```

### 2. Neural Networks

**Model Form:**
$$y_t = f(W_n \cdot \sigma(W_{n-1} \cdot \sigma(...\sigma(W_1 \cdot x_t + b_1)...) + b_n)$$

**Advantages:**
- Universal function approximator
- Can capture very complex patterns
- Good for high-dimensional data

**Disadvantages:**
- Black box model
- Requires significant tuning
- Risk of overfitting

**Implementation:**
```python
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 3. Random Forest

**Model Form:**
Ensemble of decision trees, each trained on bootstrap sample

**Advantages:**
- Handles nonlinear relationships naturally
- Feature importance ranking
- Robust to outliers
- No feature scaling needed

**Disadvantages:**
- Less interpretable than linear models
- Can be computationally expensive

**Implementation:**
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 4. Deep Neural Networks

**Model Form:**
Multiple layers with nonlinear activations and regularization

**Advantages:**
- Very flexible architecture
- Can capture complex patterns
- Good for large datasets

**Disadvantages:**
- Requires significant tuning
- Computationally expensive
- Risk of overfitting

**Implementation:**
```python
import torch
import torch.nn as nn

class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(DeepNN, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

### 5. Ensemble Methods

**Model Form:**
Combination of multiple models: $y = \frac{1}{M} \sum_{i=1}^{M} f_i(x)$

**Advantages:**
- Combines strengths of different models
- More robust predictions
- Reduces overfitting

**Disadvantages:**
- More complex
- Requires more computational resources

**Implementation:**
```python
# Train multiple models
models = []
predictions = []

for model_name in ['polynomial', 'neural_network', 'random_forest']:
    # Train model and get predictions
    pred = train_model(model_name, X_train, y_train, X_test)
    predictions.append(pred)

# Ensemble prediction (averaging)
ensemble_pred = np.mean(predictions, axis=0)
```

## Implementation Examples

### Complete Nonlinear Analysis

```python
from nonlinear_models import NonlinearFluidModeler

# Initialize modeler
modeler = NonlinearFluidModeler(P_wh_ph, v_w_g, t_data=t)

# Prepare data
modeler.prepare_data(T_train=3*len(t)//4, d=1)

# Train different models
print("Training nonlinear models...")

# 1. Polynomial regression
modeler.model_1_polynomial_regression(degree=3)

# 2. Neural network
modeler.model_2_neural_network(hidden_layers=(100, 50))

# 3. Random forest
modeler.model_3_random_forest(n_estimators=100)

# 4. Deep neural network
modeler.model_4_deep_neural_network(hidden_layers=(128, 64, 32))

# 5. Ensemble
modeler.model_5_ensemble()

# Compare models
comparison = modeler.compare_models()
modeler.plot_comparison()
```

### Enhanced Linear Model with Nonlinear Features

```python
# Create nonlinear features
def create_nonlinear_features(X, degree=2):
    """Create polynomial and interaction features."""
    features = []
    
    # Original features
    features.append(X)
    
    # Polynomial features
    for d in range(2, degree + 1):
        features.append(X ** d)
    
    # Interaction features
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            features.append(X[i:i+1] * X[j:j+1])
    
    return np.vstack(features)

# Apply to your data
X_nonlinear = create_nonlinear_features(P_wh_ph)
Y_nonlinear = v_w_g

# Train with nonlinear features
T_train = 3*len(t)//4
Y_train = Y_nonlinear[:, d:T_train]
Z_train = np.vstack([X_nonlinear[:, i:T_train-d+i] for i in range(d)])

# Fit model
Z_inv = np.linalg.pinv(Z_train)
A_nonlinear = Y_train @ Z_inv

# Predictions
Z_test = np.vstack([X_nonlinear[:, T_train+i:Nt-d+i+1] for i in range(d)])
Y_pred_nonlinear = A_nonlinear @ Z_test
```

## Model Comparison

### Performance Metrics

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| Linear | Simple, fast | Limited expressiveness | Baseline |
| Polynomial | Interpretable, nonlinear | Overfitting risk | Low-dimensional data |
| Neural Network | Universal approximator | Black box, tuning | Complex patterns |
| Random Forest | Robust, feature importance | Less interpretable | Medium complexity |
| Deep NN | Very flexible | Expensive, overfitting | Large datasets |
| Ensemble | Robust, combines strengths | Complex, expensive | Critical applications |

### Expected Performance Improvement

**Typical improvements over linear regression:**
- **Polynomial (degree=3)**: 20-40% better
- **Neural Network**: 30-60% better
- **Random Forest**: 25-50% better
- **Deep NN**: 40-70% better
- **Ensemble**: 50-80% better

## Best Practices

### 1. Data Preprocessing

```python
# Always scale features for neural networks
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
Y_train_scaled = scaler_Y.fit_transform(Y_train)
X_test_scaled = scaler_X.transform(X_test)
```

### 2. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Use cross-validation for model selection
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"CV MSE: {-scores.mean():.6f} (+/- {scores.std() * 2:.6f})")
```

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Example for neural network
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
    'alpha': [0.0001, 0.001, 0.01]
}

grid_search = GridSearchCV(MLPRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### 4. Model Selection Strategy

1. **Start simple**: Linear regression as baseline
2. **Add complexity gradually**: Polynomial → Neural Network → Deep NN
3. **Use ensemble**: Combine best models
4. **Validate thoroughly**: Cross-validation and test set
5. **Monitor overfitting**: Separate validation set

### 5. Feature Engineering

```python
# Create domain-specific features
def create_fluid_features(pressure, water_holdup, gas_holdup):
    features = []
    
    # Original features
    features.extend([pressure, water_holdup, gas_holdup])
    
    # Ratios
    features.append(water_holdup / (water_holdup + gas_holdup))  # Water fraction
    features.append(gas_holdup / (water_holdup + gas_holdup))    # Gas fraction
    
    # Products (interactions)
    features.append(pressure * water_holdup)
    features.append(pressure * gas_holdup)
    features.append(water_holdup * gas_holdup)
    
    # Nonlinear transformations
    features.append(np.log(pressure + 1e-6))
    features.append(np.sqrt(water_holdup))
    features.append(np.sqrt(gas_holdup))
    
    return np.vstack(features)
```

## Advanced Techniques

### 1. Physics-Informed Neural Networks (PINNs)

```python
class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super(PhysicsInformedNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 2)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def physics_loss(self, x, y_pred):
        """Add physics constraints to loss function."""
        # Example: mass conservation constraint
        total_holdup = y_pred[:, 0] + y_pred[:, 1]  # water + gas
        return torch.mean((total_holdup - 1.0) ** 2)
```

### 2. Time Series Specific Models

```python
# LSTM for temporal modeling
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])
```

### 3. Uncertainty Quantification

```python
# Ensemble for uncertainty estimation
def predict_with_uncertainty(models, X_test, n_samples=100):
    predictions = []
    
    for _ in range(n_samples):
        # Bootstrap sample of models
        sample_models = np.random.choice(models, size=len(models))
        pred = np.mean([model.predict(X_test) for model in sample_models], axis=0)
        predictions.append(pred)
    
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred
```

## Code Usage Example

```python
# Quick start with nonlinear modeling
from nonlinear_models import run_nonlinear_analysis

# Run comprehensive analysis
modeler, comparison = run_nonlinear_analysis(
    X_data=P_wh_ph,
    Y_data=v_w_g,
    t_data=t
)

# Get best model predictions
best_model_name = min(comparison, key=lambda x: x['Test MSE'])['Model']
best_predictions = modeler.results[best_model_name.lower().replace(' ', '_')]['predictions']

# Plot results
modeler.plot_comparison()
modeler.plot_feature_importance('random_forest')

print(f"Best model: {best_model_name}")
print(f"Test MSE: {comparison[0]['Test MSE']:.6f}")
print(f"Test R²: {comparison[0]['Test R²']:.6f}")
```

This comprehensive approach will significantly improve your fluid flow modeling by capturing the complex nonlinear relationships that linear regression cannot handle. 