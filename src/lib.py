# import numpy as np
# import matplotlib.pyplot as plt


# class DMDc:
#     def __init__(self, rank=None, p_rank=None,t=None):
#         """
#         Initialize the DMDc model.

#         :param rank: Truncated rank for SVD of X (optional)
#         :param p_rank: Truncated rank for SVD of Z (optional)
#         :param t: Time 
#         """
#         self.rank = rank
#         self.p_rank = p_rank
#         self.t = t
#         self.dt = t[1:]-t[0:-1]
        
#         # Data matrices
#         self.X = None
#         self.U = None
#         self.X_prime = None
        
#         # Learned matrices
#         self.A = None
#         self.B = None
#         self.A_tilde = None
#         self.B_tilde = None
        
#         # DMD components
#         self.modes = None
#         self.eigs = None
#         self.omega = None
#         self.dynamics = None
        
#         # SVD components
#         self.U_r = None
#         self.S_r = None
#         self.V_r = None
#         self.U_z = None
#         self.S_z = None
#         self.V_z = None
#         self.U1_z = None
#         self.U2_z = None

#     def fit(self, X, U, X_prime,t=t):
#         """
#         Fit the DMDc model to data.
#         :param X: State snapshots at time k, shape (n, m)
#         :param U: Control inputs at time k, shape (r, m)
#         :param X_prime: State snapshots at time k+1, shape (n, m)
#         """
#         self.X = X
#         self.U = U
#         self.X_prime = X_prime
#         n, m = X.shape
#         r = U.shape[0]

#         # Build augmented matrix Z = [X; U]
#         Z = np.vstack([X, U])  # shape (n + r, m)

#         # Perform truncated SVD on Z
#         U_z_full, S_z_full, Vh_z_full = np.linalg.svd(Z, full_matrices=False)
#         V_z_full = Vh_z_full.T

#         if self.p_rank is not None:
#             self.U_z = U_z_full[:, :self.p_rank].astype(np.complex128)
#             self.S_z = S_z_full[:self.p_rank].astype(np.complex128)
#             self.V_z = V_z_full[:, :self.p_rank].astype(np.complex128)
#         else:
#             self.U_z = U_z_full.astype(np.complex128)
#             self.S_z = S_z_full.astype(np.complex128)
#             self.V_z = V_z_full.astype(np.complex128)

#         # Split U_z into U1_z and U2_z
#         self.U1_z = self.U_z[:n, :]
#         self.U2_z = self.U_z[n:, :]

#         # Perform truncated SVD on X
#         U_r_full, S_r_full, Vh_r_full = np.linalg.svd(X, full_matrices=False)
#         V_r_full = Vh_r_full.T

#         if self.rank is not None:
#             self.U_r = U_r_full[:, :self.rank].astype(np.complex128)
#             self.S_r = S_r_full[:self.rank].astype(np.complex128)
#             self.V_r = V_r_full[:, :self.rank].astype(np.complex128)
#         else:
#             self.U_r = U_r_full.astype(np.complex128)
#             self.S_r = S_r_full.astype(np.complex128)
#             self.V_r = V_r_full.astype(np.complex128)

#         # Compute pseudo-inverse components
#         S_z_inv = np.diag(1 / self.S_z)

#         # Compute A and B using pseudoinverse
#         A_pinv = self.V_z @ S_z_inv @ self.U1_z.T
#         B_pinv = self.V_z @ S_z_inv @ self.U2_z.T

#         self.A = self.X_prime @ A_pinv
#         self.B = self.X_prime @ B_pinv

#         # Project A and B to lower-dimensional space
#         self.A_tilde = self.U_r.T @ self.A @ self.U_r
#         self.B_tilde = self.U_r.T @ self.B

#         # Eigendecomposition of A_tilde
#         self.eigs, W = np.linalg.eig(self.A_tilde)

#         # Compute DMD modes
#         self.modes = self.X_prime @ self.V_z @ S_z_inv @ self.U1_z.T @ self.U_r @ W

#         # Normalize modes
#         self.modes /= (np.linalg.norm(self.modes, axis=0, keepdims=True) + 1e-12)

#         # Compute continuous-time eigenvalues (frequencies)
#         self.dt = self.t[1:] - self.t[:-1]
#         print(f"Shape of t is {self.t.shape}")
#         print(f"Shape of dt is {self.dt.shape}")
        
#         mean_dt = np.mean(self.dt)
#         print(f"Shape of eigs is {self.eigs.shape}")
#         self.omega = np.log(self.eigs) / mean_dt
#         self.x_initial = self.X[:, 0]
#         self.x_initial_tilde = self.U_r.T @ self.x_initial
        


#         return self
    
#     def predict(self, X, U):
#         """
#         Predict next state using learned A and B.

#         :param X: Current states, shape (n, m)
#         :param U: Controls, shape (r, m)
#         :return: Predicted next states, shape (n, m)
#         """
#         if self.A is None or self.B is None:
#             raise ValueError("Model has not been fitted yet.")
        
#         return self.A @ X + self.B @ U

#     def predict_reduced(self, x_tilde, u):
#         """
#         Predict in reduced space using A_tilde and B_tilde.
        
#         :param x_tilde: Reduced state, shape (rank, m)
#         :param u: Control input, shape (r, m)
#         :return: Next reduced state, shape (rank, m)
#         """
#         if self.A_tilde is None or self.B_tilde is None:
#             raise ValueError("Model has not been fitted yet.")
        
#         return self.A_tilde @ x_tilde + self.B_tilde @ u

#     def project_to_reduced(self, X):
#         """
#         Project full state to reduced space.
        
#         :param X: Full state, shape (n, m)
#         :return: Reduced state, shape (rank, m)
#         """
#         if self.U_r is None:
#             raise ValueError("Model has not been fitted yet.")
        
#         return self.U_r.T @ X

#     def reconstruct_from_reduced(self, x_tilde):
#         """
#         Reconstruct full state from reduced space.
        
#         :param x_tilde: Reduced state, shape (rank, m)
#         :return: Full state, shape (n, m)
#         """
#         if self.U_r is None:
#             raise ValueError("Model has not been fitted yet.")
#         a1 = self.U_r @ x_tilde
#         a2 = np.linalg.pinv(self.U_r@self.U_r.T)
#         return a2@a1

#     def reconstruct(self):
#         """
#         Reconstruct the states using the learned model.
#         """
#         if self.X is None or self.U is None:
#             raise ValueError("Data not set.")
#         return self.predict(self.X, self.U)

#     def score(self, X=None, U=None, X_prime=None):
#         """
#         Compute relative error between true and predicted X'.

#         :param X: Optional input states
#         :param U: Optional control inputs
#         :param X_prime: Optional target states
#         :return: Relative error
#         """
#         if X is None or U is None or X_prime is None:
#             X, U, X_prime = self.X, self.U, self.X_prime
        
#         X_pred = self.predict(X, U)
#         return np.linalg.norm(X_prime - X_pred) / np.linalg.norm(X_prime)

#     def compute_dynamics(self):
#         """
#         Compute mode dynamics (time evolution coefficients).
#         """
#         if self.modes is None:
#             raise ValueError("Modes not computed yet.")

#         # Project initial state onto modes
#         print(f"Shape of X[:,0]: {self.X[:,0].shape}")
#         print(f"Shape of modes: {self.modes.shape}")
        
#         # Solve for initial amplitudes
#         try:
#             a = np.linalg.lstsq(self.modes, self.X[:, 0], rcond=None)[0]
#         except np.linalg.LinAlgError:
#             # Fallback to pseudo-inverse if lstsq fails
#             a = np.linalg.pinv(self.modes) @ self.X[:, 0]
        
#         # Time evolution
#         n_times = self.X.shape[1]
#         time_steps = np.arange(n_times)
        
#         # Create dynamics matrix
#         self.dynamics = np.zeros((len(a), n_times), dtype=complex)
        
#         for i, (amp, eig) in enumerate(zip(a, self.eigs)):
#             self.dynamics[i, :] = amp * (eig ** time_steps)
        
#         return self.dynamics

#     def simulate_future(self, x0, u_future, n_steps):
#         """
#         Simulate future states given initial condition and future controls.
        
#         :param x0: Initial state, shape (n,)
#         :param u_future: Future controls, shape (r, n_steps)
#         :param n_steps: Number of steps to simulate
#         :return: Future states, shape (n, n_steps+1)
#         """
#         if self.A is None or self.B is None:
#             raise ValueError("Model has not been fitted yet.")
        
#         n = x0.shape[0]
#         X_future = np.zeros((n, n_steps + 1))
#         X_future[:, 0] = x0
        
#         for k in range(n_steps):
#             X_future[:, k+1] = self.A @ X_future[:, k] + self.B @ u_future[:, k]
        
#         return X_future

#     def plot_modes(self, num_modes=2):
#         """
#         Plot the top DMD modes.

#         :param num_modes: Number of modes to plot
#         """
#         if self.modes is None:
#             raise ValueError("Modes not computed yet.")
        
#         num_modes = min(num_modes, self.modes.shape[1])
#         fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#         axes = axes.flatten()
        
#         for i in range(num_modes):
#             axes[i].plot(self.modes[:, i].real, label='Real', alpha=0.7)
#             axes[i].plot(self.modes[:, i].imag, label='Imag', alpha=0.7)
#             axes[i].set_title(f"Mode {i + 1}")
#             axes[i].legend()
#             axes[i].grid(True)
        
#         plt.tight_layout()
#         plt.show()

#     def plot_dynamics(self, num_modes=6):
#         """
#         Plot the mode dynamics.
#         """
#         if self.dynamics is None:
#             self.compute_dynamics()
        
#         num_modes = min(num_modes, self.dynamics.shape[0])
#         plt.figure(figsize=(12, 6))
        
#         for i in range(num_modes):
#             plt.subplot(2, 1, 1)
#             plt.plot(self.dynamics[i, :].real, label=f"Mode {i + 1}")
#             plt.title("Mode Dynamics (Real)")
#             plt.legend()
#             plt.grid(True)
#             plt.yscale("log")
            
#             plt.subplot(2, 1, 2)
#             plt.plot(self.dynamics[i, :].imag, label=f"Mode {i + 1}")
#             plt.title("Mode Dynamics (Imaginary)")
#             plt.legend()
#             plt.grid(True)
#             plt.yscale("log")
        
#         plt.tight_layout()
#         plt.show()

#     def plot_eigenvalues(self):
#         """
#         Plot eigenvalues in complex plane.
#         """
#         if self.eigs is None:
#             raise ValueError("Eigenvalues not computed yet.")
        
#         plt.figure(figsize=(8, 8))
#         plt.scatter(self.eigs.real, self.eigs.imag, marker='x', s=100, alpha=0.7)
        
#         # Add unit circle
#         theta = np.linspace(0, 2*np.pi, 100)
#         plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit Circle')
        
#         plt.axhline(0, color='black', lw=0.5)
#         plt.axvline(0, color='black', lw=0.5)
#         plt.xlabel('Real')
#         plt.ylabel('Imaginary')
#         plt.title('Eigenvalues in Complex Plane')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.axis('equal')
#         plt.show()

#     def plot_svd(self):
#         """
#         Plot singular values from SVD.
#         """
#         if self.S_r is None or self.S_z is None:
#             raise ValueError("Singular values not computed yet.")
        
#         plt.figure(figsize=(12, 5))
        
#         plt.subplot(1, 2, 1)
#         plt.semilogy(self.S_r, 'o-', alpha=0.7)
#         plt.title('Singular Values of X')
#         plt.xlabel('Index')
#         plt.ylabel('Singular Value')
#         plt.grid(True)
        
#         plt.subplot(1, 2, 2)
#         plt.semilogy(self.S_z, 's-', alpha=0.7, color='red')
#         plt.title('Singular Values of Augmented Z')
#         plt.xlabel('Index')
#         plt.ylabel('Singular Value')
#         plt.grid(True)
        
#         plt.tight_layout()
#         plt.show()

#     def plot_reconstruction(self, idx=0):
#         """
#         Plot original vs reconstructed state over time.

#         :param idx: Index of the feature/state to visualize
#         """
#         X_recon = self.reconstruct()
        
#         plt.figure(figsize=(12, 6))
#         plt.plot(self.X[idx, :], 'o-', label="Original", alpha=0.7, markersize=1)
#         plt.plot(X_recon[idx, :], 's--', label="Reconstructed", alpha=0.7, markersize=1)
#         plt.title(f"State Comparison (Feature {idx})")
#         plt.xlabel('Time Step')
#         plt.ylabel('State Value')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
        
#         # Also show error
#         error = np.abs(self.X[idx, :] - X_recon[idx, :])
#         plt.figure(figsize=(10, 4))
#         plt.plot(error, 'r-', alpha=0.7)
#         plt.title(f"Reconstruction Error (Feature {idx})")
#         plt.xlabel('Time Step')
#         plt.ylabel('Absolute Error')
#         plt.grid(True)
#         plt.show()