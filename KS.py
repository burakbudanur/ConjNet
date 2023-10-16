import numpy as np
from scipy.integrate import odeint, simpson
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, identity, sqrt, sign
from numpy.linalg import inv, norm
import json
import os
import h5py
from pathlib import Path

class KS():

    def __init__(self,L, K, dt, atol=1e-12, rtol=1e-12):

        self.L = L
        self.K = K
        self.atol = atol
        self.rtol = rtol
        self.dt = dt
        self.refredcoefs = np.loadtxt(Path(__file__).parent / "opt_coefs.dat")

        d = 2 * K
        self.d = d

        self.reflection_matrix = np.identity(d)
        for q in range(0, d, 2):
            self.reflection_matrix[q, q] = -1

        self.SO2_generator = np.zeros((d, d))

        for q in range(1, d//2 + 1):
            offset = 2 * (q - 1)
            self.SO2_generator[offset:offset+2, offset:offset+2] = np.array(
                [[0, -q],
                [q,  0]]
                )

        self.dv_dxi = np.zeros((self.d,self.d), dtype=complex)
        self.dxi_dv = np.zeros((self.d,self.d), dtype=complex)
        for m in range(-self.d//2, self.d//2 + 1):
                    
            if m == 0: continue
            elif m > 0: mm = m - 1
            else: mm = m

            for l in range(1, self.d+1):
        
                ll = l - 1 # l: vector index, ll: array index
                if m>0:
                    if 2 * (m - 1) + 1 == l:
                        self.dv_dxi[mm, ll] += 1
                    if 2 * (m - 1) + 2 == l:
                        self.dv_dxi[mm, ll] += 1j
                    
                else:
                    if 2 * (- m - 1) + 1 == l:
                        self.dv_dxi[mm, ll] += 1
                    if 2 * (- m - 1) + 2 == l:
                        self.dv_dxi[mm, ll] -= 1j

                if l%2 == 1:
                    if (l - 1) / 2 + 1 == m or (l - 1) / 2 + 1 == - m:
                        self.dxi_dv[ll, mm] += 0.5
                else:
                    if (l - 2) / 2 + 1 == m: 
                        self.dxi_dv[ll, mm] += 0.5 / 1j
                    if (l - 2) / 2 + 1 == - m:
                        self.dxi_dv[ll, mm] -= 0.5 / 1j

    def rhs(self, xi, t):
        """
        RHS of the Kuramoto--Sivashinsky equation
        u_t = - u_xx - u_xxxx - 0.5 (u^2)_x 
        
        Parameters
        ----------
        xi: np.array
            [a_1, b_1, a_2, b_2, ..., a_{d/2} b_{d/2}] 
            where a_i = Re v_i, b_i = Im v_i, v = Fourier{u}
        t: float, unused
            time variable for compatibility with scipy.integrate.odeint

        Returns
        -------
        \d xi / \d t = rho(xi) derived from the RHS of the KS equation
        """
        
        L = self.L 
        d = self.d 
        
        N = d + 2 # num. grid points
        v = np.zeros(d//2 + 2, dtype=complex) 
        v[1:d//2 + 1] = xi[0:d:2] + 1j * xi[1:d:2] 

        k = np.arange(0, d//2 + 2) * (2 * pi / L) # wave numbers

        # Pseudospectral nonlinear term
        u = np.fft.irfft(v, n=N) # must specify the grid size for irfft
        n = 0.5 * 1j * k * N * np.fft.rfft(u * u) # N: fft normalization
        # n = Fourier{0.5 u_x^2}
        
        vdot = (k ** 2 - k ** 4) * v - n 
        xidot = np.zeros(xi.shape)
        xidot[0:d:2] = np.real(vdot[1:d//2 + 1])
        xidot[1:d:2] = np.imag(vdot[1:d//2 + 1])

        return xidot

    def flow(self, xi, t):
        """
        Finite time flow. Returns 
            xi_t = xi + \int_0^t xi(t') dt', where xi(0) = xi
        """

        dt = self.dt
        L = self.L
        if t < dt:
            tt = np.array([0, t])
        else:
            tt = np.linspace(0, t, int(t / dt) + 1)

        xi_t = odeint(
            self.rhs, xi, tt, atol = self.atol, rtol = self.rtol
            )[-1, :]
        return xi_t

    def plot_spacetime(self, solution, tt, ax = None, vmin = None, vmax = None):
        """
        Space-time plot of the solution.

        Parameters
        ----------
        solution: np.array (nt x d)
            nt x d solution matrix with xi(t(0)), xi(t(1)), ... in its rows
        tt: np.array (nt)
            time points
        """
        if ax is None:
            fig = plt.figure(figsize=(2, 6))
            ax  = fig.gca()
        else:
            fig = plt.gcf()
        
        L = self.L
        nt = solution.shape[0]
        d = solution.shape[1]
        N = d + 2 # num. grid points
        x = np.linspace(0, L, N)
        
        vv = np.zeros((nt, d//2 + 2), dtype=complex) 
        vv[: , 1:d//2 + 1] = solution[:, 0:d:2] + 1j * solution[:, 1:d:2] 
        # v[0] (mod 0) = v[d//2 + 2] (Re mod d//2+1) = 0

        uu = np.fft.irfft(vv, n=N) # must specify the grid size for irfft

        im = ax.pcolormesh(x, tt, uu, 
                        cmap='RdBu', shading='gouraud', 
                        rasterized=True, vmin=vmin, vmax=vmax)

        ax.set_xlabel("x")
        ax.set_ylabel("t")
        
        return im, fig, ax
    
    def SO2_operator(self, theta):
        """
        Return block diagonal d x d SO(2) matrix
        """

        d = self.d
        g = np.zeros((d, d))

        for q in range(1, d//2 + 1):
            offset = 2 * (q - 1)
            g[offset:offset + 2, offset:offset + 2] = np.array(
                [[cos(q * theta), -sin(q * theta)],
                [sin(q * theta), cos(q * theta)]]
                )

        return g
    def symmreduce(self, sol):
        
        if sol.ndim == 1:
            sol = np.array([sol]) # making it 1 x d 

        d = self.d 
        K = self.K
        opt_coefs = self.refredcoefs
        
        compl_sol = sol[:, 0:d:2] + 1j * sol[:, 1:d:2]
        slice_phase = (np.angle(compl_sol[:, 1]) - np.pi/2) / 2
        compl_sol_ = compl_sol.copy()

        for ik, k in enumerate(range(1,d//2+1)):
            compl_sol_[:, ik] *= np.exp(- 1j * k * slice_phase)

        i_even_modes = np.arange(1, d//2, 2)
        i_odd_modes = np.arange(0, d//2, 2)
        
        vhat_1 = compl_sol_[:, i_odd_modes[0]] 
        phipi = np.angle(vhat_1)
        compl_sol__ = compl_sol_.copy()
        compl_sol__[:, i_odd_modes] *= np.exp(1j * phipi).reshape(
            compl_sol_.shape[0], 1
            ) # reducing pi-shift symmetry

        compl_red_sol = compl_sol__.copy()

        compl_ref_modes = np.zeros((sol.shape[0], (K-1) // 2), dtype=complex)
        compl_inv_modes = np.zeros((sol.shape[0], (K-1) // 2 + 1), dtype=complex)

        # projecting the reflection-equvariant state vector into symm/antisymm subspaces

        compl_inv_modes[:, 0] = compl_sol__[:, i_even_modes[0]]
        compl_inv_modes[:, 1:1+len(i_even_modes[1:])] = (
            np.imag(compl_sol__[:, i_even_modes[1:]])
            + 1j * np.real(compl_sol__[:, i_odd_modes[0:len(i_even_modes[1:])]])
        )
        compl_inv_modes[:, -1] = (
            np.real(compl_sol__[:, i_odd_modes[-2]]) 
            + 1j * np.real(compl_sol__[:, i_odd_modes[-1]])
        )

        compl_ref_modes[:, 0:len(i_even_modes[1:])] = (
            np.real(compl_sol__[:, i_even_modes[1:]])
            + 1j * np.imag(compl_sol__[:, i_odd_modes[0:len(i_even_modes[1:])]])
        )
        compl_ref_modes[:, -1] = (
            np.imag(compl_sol__[:, i_odd_modes[-2]]) 
            + 1j * np.imag(compl_sol__[:, i_odd_modes[-1]])
        )

        coefs = opt_coefs[0::2] + 1j * opt_coefs[1::2]
        symmred_mode = compl_ref_modes @ coefs
        symmred_phase = np.angle(symmred_mode)

        compl_red_sol[:, 0:(K-1)//2+1] = compl_inv_modes
        compl_red_sol[:, (K-1)//2+1:] = (
            compl_ref_modes 
            * np.exp(1j * symmred_phase).reshape(compl_ref_modes.shape[0], 1)
        )
        red_sol = sol.copy()
        red_sol[:, 0::2] = np.real(compl_red_sol)
        red_sol[:, 1::2] = np.imag(compl_red_sol)

        if (red_sol.shape[0] == 1): 
            return red_sol.reshape(-1)
        else: 
            return red_sol

    def inv_symmreduce(self, red_sol):
        
        if red_sol.ndim == 1:
            red_sol = np.array([red_sol]) # making it 1 x d 

        K = self.K 
        d = self.d 
        opt_coefs = self.refredcoefs
        coefs = opt_coefs[0::2] + 1j * opt_coefs[1::2]


        # inverting the 
        compl_red_sol = red_sol[:, 0::2] + 1j * red_sol[:, 1::2]
        compl_inv_modes = compl_red_sol[:, 0:(K-1)//2+1]
        compl_refred_modes = compl_red_sol[:, (K-1)//2+1:]

        symmred_modered = compl_refred_modes @ coefs
        symmred_phase = np.angle(symmred_modered) / 2.0 
        

        compl_ref_modes = (
            compl_refred_modes 
            * np.exp(-1j * symmred_phase).reshape(compl_refred_modes.shape[0], 1)
            )
        
        compl_sol__ = np.zeros(compl_red_sol.shape, dtype=complex)
        i_even_modes = np.arange(1, d//2, 2)
        i_odd_modes = np.arange(0, d//2, 2)

        compl_sol__[:, i_even_modes[0]] = compl_inv_modes[:, 0]
        compl_sol__[:, i_even_modes[1:]] += 1j * np.real(
            compl_inv_modes[:, 1:1+len(i_even_modes[1:])]
            )
        compl_sol__[:, i_odd_modes[0:len(i_even_modes[1:])]] += np.imag(
            compl_inv_modes[:, 1:1+len(i_even_modes[1:])]
        )
        compl_sol__[:, i_odd_modes[-2]] += np.real(compl_inv_modes[:, -1])
        compl_sol__[:, i_odd_modes[-1]] += np.imag(compl_inv_modes[:, -1])
        compl_sol__[:, i_even_modes[1:]] += np.real(
            compl_ref_modes[:, 0:len(i_even_modes[1:])]
            )
        compl_sol__[:, i_odd_modes[0:len(i_even_modes[1:])]] += 1j * np.imag(
            compl_ref_modes[:, 0:len(i_even_modes[1:])]
        )
        compl_sol__[:, i_odd_modes[-2]] += 1j * np.real(compl_ref_modes[:, -1])
        compl_sol__[:, i_odd_modes[-1]] += 1j * np.imag(compl_ref_modes[:, -1])

        vhat_1red = compl_sol__[:, i_odd_modes[0]]
        phipi = np.angle(vhat_1red) / 2

        compl_sol_ = compl_sol__.copy()
        compl_sol__[:, i_odd_modes] *= np.exp(- 1j * phipi).reshape(
            compl_sol_.shape[0], 1
            ) # reverting pi-reduction
        
        compl_sol = compl_sol__

        sol = np.zeros((compl_sol.shape[0],d))
        sol[:, 0:d:2] = np.real(compl_sol)
        sol[:, 1:d:2] = np.imag(compl_sol)

        if (sol.shape[0] == 1): 
            return sol.reshape(-1)
        else: 
            return sol
        

    def reconst_sol(self, sol_, tt, alpha_th = 0.9, xi_0 = None, full_output=False, verbose=False):
        # Reconstruction of the symmetry-reduced solution

        d = self.d
        slice_temp = np.zeros(d)
        slice_temp[3] = 1.0 

        inv_sol_ = self.inv_symmreduce(sol_)
        
        try:
            # aligning the initial state with 
            cxi_0 = xi_0[0:d:2] + 1j * xi_0[1:d:2]
            slice_phase_0 = (np.angle(cxi_0[1]) - pi/2) / 2 
            inv_sol_0 = inv_sol_[0, :]
            
            ref_inv_sol_0 = self.reflection_matrix @ inv_sol_0

            inv_sol_0_shift = self.SO2_operator(slice_phase_0) @ inv_sol_0
            inv_sol_0_shift_ = self.SO2_operator(slice_phase_0 + pi) @ inv_sol_0
            ref_inv_sol_0_shift = self.SO2_operator(slice_phase_0) @ ref_inv_sol_0
            ref_inv_sol_0_shift_ = self.SO2_operator(slice_phase_0 + pi) @ ref_inv_sol_0
            
            dists = np.zeros(4)
            dists[0] = norm(inv_sol_0_shift - xi_0)
            dists[1] = norm(inv_sol_0_shift_ - xi_0)
            dists[2] = norm(ref_inv_sol_0_shift - xi_0)
            dists[3] = norm(ref_inv_sol_0_shift_ - xi_0)
            
            mindist = np.min(dists)
            
            if mindist == dists[0]:
                reflect = False
            elif mindist == dists[1]:
                reflect = False
                slice_phase_0 += pi
            elif mindist == dists[2]:
                reflect = True
            else:
                reflect=True
                slice_phase_0 += pi
                
        except Exception as e: 
            print(e)
            slice_phase_0 = 0
            reflect = False

        def get_alpha(sol):
            """
            get an estimate of the state space velocity between consecutive 
            time steps
            """
            dt_sol = np.gradient(sol, axis=0)
            alpha = np.zeros(dt_sol.shape[0] - 1)
            alpha[:] = np.sum(dt_sol[1:, :] * dt_sol[:-1, :], axis=1) / (
                norm(dt_sol[1:, :], axis=1) * norm(dt_sol[:-1, :], axis=1)
            )
            return alpha        
        
        rec_sol_slice = inv_sol_.copy()
        alpha_rec = get_alpha(rec_sol_slice)
        disconts = np.argwhere(alpha_rec < alpha_th).reshape(-1)

        disc_symms = [
            self.reflection_matrix, 
            self.SO2_operator(np.pi), 
            self.reflection_matrix @ self.SO2_operator(np.pi)
            ]
        
        n_failed = 0 

        while len(disconts) - n_failed > 0:
            n = disconts[n_failed]
            rec_failed = False
            k_max_alpha, max_alpha = (-1, alpha_rec[n])

            for k, si in enumerate(disc_symms):
                alt_sol = rec_sol_slice.copy()
                alt_sol[n+2:, :] = alt_sol[n+2:, :] @ si.transpose()
                
                alpha_alt = get_alpha(alt_sol)
                alt_disconts = np.argwhere(alpha_alt < alpha_th).reshape(-1)
                
                if verbose:
                    print(k, alt_disconts[0:4], get_alpha(alt_sol)[alt_disconts[0:4]])

                if not(n in alt_disconts):
                    rec_sol_slice = alt_sol.copy()
                    disconts = alt_disconts.copy()
                    break
                elif alpha_alt[n] > max_alpha:
                    max_alpha = float(alpha_alt[n])
                    k_max_alpha = int(k)
                
            if n in alt_disconts and k == 2:
                print(f"alpha_th condition not met at n = {n}")
                print("picking the smoothest alternative...")
                n_failed += 1 
                if not(k_max_alpha == -1):
                    si = disc_symms[k_max_alpha]
                    rec_sol_slice[n+2:, :] = rec_sol_slice[n+2:, :] @ si.transpose()



        def slice_phase_vel(xi_hat, slice_temp = slice_temp):
            slice_tan = self.SO2_generator @ slice_temp
            tan_xi_hat = self.SO2_generator @ xi_hat
            return (self.rhs(xi_hat, 0) @ slice_tan) / (tan_xi_hat @ slice_tan) 

        rhs_phi = np.array(
            list(
                    map(
                        lambda n: slice_phase_vel(rec_sol_slice[n], slice_temp=slice_temp), 
                        range(rec_sol_slice.shape[0])
                        )
                )
            )
        rec_sol = rec_sol_slice.copy()
        for n in range(1,len(tt)):
            phi = simpson(rhs_phi[0:n], tt[0:n])
            rec_sol[n, :] = self.SO2_operator(phi) @ rec_sol_slice[n, :]    

        if reflect:
            rec_sol = (self.reflection_matrix @ rec_sol.transpose()).transpose() 

        rec_sol = (self.SO2_operator(slice_phase_0) @ rec_sol.transpose()).transpose()

        if not(full_output):
            return rec_sol
        else:
            return rec_sol, slice_phase_0, reflect
        
        
    def power_in(self, solution):
        
        L = self.L
        single_state = False
        if solution.ndim == 1:
            single_state = True
            solution = np.array([solution])

        nt = solution.shape[0]
        d = solution.shape[1]
        k = np.arange(0, d//2 + 2) * (2 * pi / L) # wave numbers

        ikvv = np.zeros((nt, d//2 + 2), dtype=complex) 
        ikvv[: , 1:d//2 + 1] = solution[:, 0:d:2] + 1j * solution[:, 1:d:2]
        ikvv = 1j * k * ikvv

        P = np.sum(np.abs(ikvv) ** 2, axis=1)

        if single_state: 
            return P[0]
        else: 
            return P


    def dissipation(self, solution):
        
        L = self.L
        single_state = False
        if solution.ndim == 1:
            single_state = True
            solution = np.array([solution])

        nt = solution.shape[0]
        d = solution.shape[1]
        k2 = (np.arange(0, d//2 + 2) * (2 * pi / L)) ** 2 # wave numbers

        ik2vv = np.zeros((nt, d//2 + 2), dtype=complex) 
        ik2vv[: , 1:d//2 + 1] = solution[:, 0:d:2] + 1j * solution[:, 1:d:2]
        ik2vv = -1 * k2 * ik2vv

        D = np.sum(np.abs(ik2vv) ** 2, axis=1)

        if single_state: 
            return D[0]
        else: 
            return D
