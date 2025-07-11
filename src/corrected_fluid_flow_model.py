import numpy as np
import matplotlib.pyplot as plt
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class fluid_flow_model:
    def __init__(self, P_wh_ph, v_w_g, gpu=False, d=1, T_train=None):
        self.P_wh_ph = P_wh_ph
        self.v_w_g = v_w_g
        self.gpu = gpu and CUPY_AVAILABLE  # Only use GPU if cupy is available
        self.d = d
        self.P_wh_ph_gpu = None
        self.v_w_g_gpu = None
        self.A = None
        self.b = None
        self.A_gpu = None
        self.b_gpu = None
        self.T_train = T_train
        self.Z_full_gpu = None
        self.Z_test_gpu = None
        self.Y_pred_full_gpu = None
        self.Y_pred_test_gpu = None
        self.Y_test = None
        self.Y_full = None
        self.Y_pred_full = None
        self.Y_pred_test = None
        self.Z_full = None
        self.Z_test = None
        
    def model_1_linear_regression(self):
        self.Z = self.P_wh_ph
        self.Y = self.v_w_g
        self.Y_train = self.Y[:,self.d:self.T_train]
        self.Z_train = np.vstack([self.Z[:,i:self.T_train-self.d+i] for i in range(self.d)])
        self.Z_train = np.vstack([self.Z_train, np.ones((self.Z_train.shape[1]))])
        print(f"Z_train shape: {self.Z_train.shape}\n")
        print(f"Y_train shape: {self.Y_train.shape}\n")
        print(f"Z_train shape: {self.Z_train.shape}\n")

        if self.gpu:
            P_wh_ph_gpu = cp.array(self.P_wh_ph)
            v_w_g_gpu = cp.array(self.v_w_g)
            Z_train_gpu = cp.array(self.Z_train)
            Y_train_gpu = cp.array(self.Y_train)
            Ab_gpu  = cp.linalg.lstsq(Z_train_gpu.T, Y_train_gpu.T, rcond=None)[0].T 
            self.A = Ab_gpu.copy()[:,:-1]
            self.b = Ab_gpu.copy()[:,-1]
            self.A_gpu = cp.array(self.A)
            self.b_gpu = cp.array(self.b)
        else:
            Ab = np.linalg.lstsq(self.Z_train.T, self.Y_train.T, rcond=None)[0].T 
            self.A = Ab[:,:-1]
            self.b = Ab[:,-1].reshape(-1,1)
        print(f"A shape: {self.A.shape}\nb shape: {self.b.shape}\n")
        
        # Initialize Z_full and Z_test for prediction
        Nt = self.Z.shape[1]  # Total number of time steps
        self.Z_full = np.vstack([self.Z[:,i:Nt-self.d+i+1] for i in range(self.d)])
        self.Z_test = np.vstack([self.Z[:,self.T_train+i:Nt-self.d+i+1] for i in range(self.d)])
        
        if self.gpu:
            self.Z_full_gpu = cp.array(self.Z_full)
            self.Z_test_gpu = cp.array(self.Z_test)
            
        return self.A, self.b
        
    def predict(self):
        if self.gpu:
            Y_pred_full_gpu = self.A_gpu @ self.Z_full_gpu + self.b_gpu
            Y_pred_test_gpu = self.A_gpu @ self.Z_test_gpu + self.b_gpu
            self.Y_pred_full = Y_pred_full_gpu.copy()
            self.Y_pred_test = Y_pred_test_gpu.copy()
        else:
            self.Y_pred_full = self.A @ self.Z_full + self.b
            self.Y_pred_test = self.A @ self.Z_test + self.b
        return self.Y_pred_full, self.Y_pred_test
        
    def relative_error(self):
        self.Y_test = self.Y[:,self.T_train-self.d:self.T_train]
        self.Y_full = self.Y[:,self.d:self.T_train]
        
        # Calculate predictions if not already done
        if self.Y_pred_full is None or self.Y_pred_test is None:
            self.predict()
            
        self.accuracy = np.linalg.norm(self.Y_pred_full - self.Y_full)/np.linalg.norm(self.Y_full)
        self.accuracy_test = np.linalg.norm(self.Y_pred_test - self.Y_test)/np.linalg.norm(self.Y_test)
        return self.accuracy, self.accuracy_test
        
    def plot(self):
        plt.figure(figsize=(10,10))
        plt.subplot(2, 2, 1)
        plt.title(f"Y_pred_full,T_train={self.T_train}, d={self.d}")
        plt.pcolormesh(self.Y_pred_full,cmap='jet',label='Y_pred_full', vmin=np.percentile(self.Y_pred_full,1), vmax=np.percentile(self.Y_pred_full,99)) # vmin and vmax are used to normalize the data, they are the percentiles of the data
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.title("Y_full")
        plt.pcolormesh(self.Y,cmap='jet',label='Y_full')
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.title(f"Y_pred_test,T_train={self.T_train}, d={self.d}")
        plt.pcolormesh(self.Y_pred_test,cmap='jet',label='Y_pred_test', vmin=np.percentile(self.Y_pred_full,1), vmax=np.percentile(self.Y_pred_full,99)) # vmin and vmax are used to normalize the data, they are the percentiles of the data
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.title("Y_test")
        plt.pcolormesh(self.Y_test,cmap='jet',label='Y_test')
        plt.colorbar()
        plt.show()

# Example usage:
# model = fluid_flow_model(P_wh_ph, v_w_g, gpu=False, d=1, T_train=3*Nt//4)
# model.model_1_linear_regression()
# model.predict()
# model.relative_error()
# model.plot()
# print(f"Accuracy: {model.accuracy}\nAccuracy_test: {model.accuracy_test}\n") 