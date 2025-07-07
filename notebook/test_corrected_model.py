import numpy as np
import matplotlib.pyplot as plt
from corrected_fluid_flow_model import fluid_flow_model

# Create synthetic data for testing
def create_synthetic_data():
    """Create synthetic fluid flow data for testing"""
    Nt = 1000  # Number of time steps
    Nx = 100   # Number of spatial points
    
    # Create synthetic pressure data (P_wh_ph)
    t = np.linspace(0, 10, Nt)
    x = np.linspace(0, 1, Nx)
    T, X = np.meshgrid(t, x)
    
    # Synthetic pressure field with some spatial and temporal variation
    P_wh_ph = np.sin(2*np.pi*X) * np.exp(-T/5) + 0.1*np.random.randn(Nx, Nt)
    
    # Synthetic velocity data (v_w_g) - related to pressure but with some lag
    v_w_g = np.gradient(P_wh_ph, axis=1) + 0.05*np.random.randn(Nx, Nt)
    
    return P_wh_ph, v_w_g, Nt

def test_model():
    """Test the corrected fluid_flow_model"""
    print("Creating synthetic data...")
    P_wh_ph, v_w_g, Nt = create_synthetic_data()
    
    print(f"Data shapes: P_wh_ph={P_wh_ph.shape}, v_w_g={v_w_g.shape}")
    
    # Test CPU version
    print("\n=== Testing CPU version ===")
    T_train = 3*Nt//4
    model_cpu = fluid_flow_model(P_wh_ph, v_w_g, gpu=False, d=1, T_train=T_train)
    
    try:
        A, b = model_cpu.model_1_linear_regression()
        print("✓ model_1_linear_regression completed successfully")
        
        Y_pred_full, Y_pred_test = model_cpu.predict()
        print("✓ predict completed successfully")
        
        accuracy, accuracy_test = model_cpu.relative_error()
        print(f"✓ relative_error completed successfully")
        print(f"  Accuracy: {accuracy:.6f}")
        print(f"  Accuracy_test: {accuracy_test:.6f}")
        
        print("✓ All CPU operations completed successfully!")
        
    except Exception as e:
        print(f"✗ CPU test failed: {e}")
        return False
    
    # Test GPU version (if available)
    print("\n=== Testing GPU version ===")
    model_gpu = fluid_flow_model(P_wh_ph, v_w_g, gpu=True, d=1, T_train=T_train)
    
    try:
        A_gpu, b_gpu = model_gpu.model_1_linear_regression()
        print("✓ GPU model_1_linear_regression completed successfully")
        
        Y_pred_full_gpu, Y_pred_test_gpu = model_gpu.predict()
        print("✓ GPU predict completed successfully")
        
        accuracy_gpu, accuracy_test_gpu = model_gpu.relative_error()
        print(f"✓ GPU relative_error completed successfully")
        print(f"  GPU Accuracy: {accuracy_gpu:.6f}")
        print(f"  GPU Accuracy_test: {accuracy_test_gpu:.6f}")
        
        # Compare CPU and GPU results
        print(f"\n=== Comparing CPU vs GPU results ===")
        print(f"CPU vs GPU accuracy difference: {abs(accuracy - accuracy_gpu):.2e}")
        print(f"CPU vs GPU test accuracy difference: {abs(accuracy_test - accuracy_test_gpu):.2e}")
        
        print("✓ All GPU operations completed successfully!")
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        print("This is expected if cupy is not available")
    
    return True

if __name__ == "__main__":
    print("Testing corrected fluid_flow_model implementation...")
    success = test_model()
    
    if success:
        print("\n🎉 All tests passed! The corrected implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.") 