# These are required
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

# This is only necessary to load the previously prepared data
from torch import load

# This is the class managing the simulation.
class BurgersROM:

    def __init__(self):
        pass


    def POD_from_data(self, file, n):
        '''
        Loads snapshot data from my BurgersData file.
        Computes POD approximation after subtracting initial condition.
        '''
        data = load(file)
        S = data['S']
        M = data['M']

        tStart = perf_counter()
        for i in range(12):
            IC = np.ones(S.shape[1])
            IC[0] = M[101*i][1]
            S[101*i:101*(i+1)] = S[101*i:101*(i+1)] - IC.reshape(1,-1)
        self.POD = np.linalg.svd(S.T)[0][:,:n]
        tEnd = perf_counter()

        print(f'POD time is {tEnd-tStart}')


    def POD_from_parameters(self, N, Nt, t_max, mu_list, n):
        '''
        Computes a POD approximation from a user-defined list of parameters.
        '''
        sols = [0 for mu in mu_list]
        u0 = np.ones(N)
        for i, mu in enumerate(mu_list):
            sols[i] = self.solve_FOM(N, Nt, t_max, mu, verbose=False)
            u0[0] = mu[0]
            sols[i] = sols[i] - u0.reshape(-1,1)
        snapshots = np.concatenate(sols, axis=1)
        self.POD = np.linalg.svd(snapshots)[0][:,:n]


    def solve_FOM(self, N, Nt, t_max, mu, I=[0,100], verbose=True, slow=False):
        '''
        Computes the (N x Nt) FOM solution at parameter mu.
        '''
        def f(u): return 0.5 * u**2  # Convenience function

        # Define parameters
        a, b = I
        dt = t_max / Nt
        x = np.linspace(a, b, N)
        dx = (b - a) / N

        # Define forcing.
        g = 0.02 * np.exp(mu[1] * x)

        if slow:
            def tridiag(a, b, c, k1=-1, k2=0, k3=1):
                return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

            fdiag = np.insert(np.ones(N-1), 0, 0)
            FD = tridiag(-np.ones(N-1), fdiag, np.zeros(N-1)) / dx

        # Set up the initial solution values.
        u0 = np.ones_like(x)
        u0[0] = mu[0]  # BC at left endpoint

        # Initialize quantities.
        U = np.zeros((Nt+1, N))
        U[0] = u0

        # Implementation of the numerical method.
        self.FOMtElapsed = 0
        for i in range(1, Nt+1):

            # Upwind conservative.
            FOMtStart = perf_counter()
            u = U[i-1]
            uNew = u
            if slow:
                uNew[1:] = u[1:] + dt * (-1 * FD[1:] @ f(u) + g[1:])
            else:
                uNew[1:] = u[1:] + dt * (-1 / dx * (f(u[1:]) - f(u[:-1])) + g[1:])

            FOMtEnd = perf_counter()
            self.FOMtElapsed = self.FOMtElapsed + FOMtEnd - FOMtStart

            # Save the latest result.
            U[i] = uNew

        if verbose: print(f'FOM time is {self.FOMtElapsed}')
        return U.T


    def assemble(self, mu, I=[0,100], verbose=True):
        '''
        Assembles ROM system at parameter mu.
        Uses POD basis computed earlier.
        '''
        a, b = I
        N = self.POD.shape[0]
        x = np.linspace(a, b, N)
        dx = (b - a) / N

        def tridiag(a, b, c, k1=-1, k2=0, k3=1):
            return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

        tStart = perf_counter()

        # Precompute quantities for ROM.
        PODinv = self.POD.T  # Since POD comes from unitary mtx
        self.gRed = PODinv @ (0.02 * np.exp(mu[1] * x))

        fdiag = np.insert(np.ones(N-1), 0, 0)
        FD = tridiag(-np.ones(N-1), fdiag, np.zeros(N-1)) / dx
        fd = PODinv @ FD  # low-dim FD matrix

        u0 = np.ones_like(x)
        u0[0] = mu[0]  # BC at left endpoint

        self.aVec = fd @ (0.5*u0**2)  # rank-1 term in (u^2)_x

        self.Bmat = fd @ np.diag(u0) @ self.POD  # rank-2 term in (u^2)_x

        outprod = np.tensordot(self.POD, self.POD, axes=0)
        temp = 0.5 * outprod.diagonal(axis1=0, axis2=2).transpose((2,1,0))
        self.Ctens = np.einsum('ij,jkl', fd, temp)  # rank-3 term in (u^2)_x

        tEnd = perf_counter()
        if verbose:  print(f'assembly time is {tEnd-tStart}')


    def solve_ROM(self, Nt, t_max, I=[0,100], verbose=True):
        '''
        Forward Euler to solve the pre-assembled ROM.
        '''
        # Define parameters
        a, b = I
        dt = t_max / Nt
        N = self.POD.shape[0]
        x = np.linspace(a, b, N)
        dx = (b-a) / N

        # Initialize quantities.
        uRed0 = np.zeros(self.POD.shape[1])
        U = np.zeros((Nt+1, self.POD.shape[1]))
        U[0] = uRed0

        # Implementation of the numerical method.
        self.ROMtElapsed = 0
        for i in range(1,Nt+1):
            ROMtStart = perf_counter()

            uRed = U[i-1]
#             uRed2 = np.tensordot(uRed, uRed, axes=0)
#             uRedNew = uRed + dt * (self.gRed - self.aVec - self.Bmat @ uRed \
#                                    - np.einsum('ijk,jk', self.Ctens, uRed2))

            uRedNew = uRed + dt * (self.gRed - self.aVec - self.Bmat @ uRed \
                                   - (self.Ctens @ uRed) @ uRed)

            ROMtEnd = perf_counter()
            self.ROMtElapsed = self.ROMtElapsed + ROMtEnd - ROMtStart

            #  Save the latest result.
            U[i] = uRedNew

        if verbose:  print(f'ROM time is {self.ROMtElapsed}')
        return U.T


    def reconstruct(self, approx, mu):
        '''
        Reconstruct approximation to the FOM solution at mu from approx.
        '''
        u0 = np.ones(self.POD.shape[0])
        u0[0] = mu[0]
        return u0.reshape(-1,1) + self.POD @ approx
