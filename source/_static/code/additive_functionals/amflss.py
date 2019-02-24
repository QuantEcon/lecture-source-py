""" 
@authors: Chase Coleman, Balint Szoke, Tom Sargent

"""

import numpy as np
import scipy as sp
import scipy.linalg as la
import quantecon as qe
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm


class AMF_LSS_VAR:
    """
    This class transforms an additive (multiplicative)
    functional into a QuantEcon linear state space system.
    """

    def __init__(self, A, B, D, F=None, ν=None):
        # Unpack required elements
        self.nx, self.nk = B.shape
        self.A, self.B = A, B

        # checking the dimension of D (extended from the scalar case)
        if len(D.shape) > 1 and D.shape[0] != 1:
            self.nm = D.shape[0]
            self.D = D
        elif len(D.shape) > 1 and D.shape[0] == 1:
            self.nm = 1
            self.D = D
        else:
            self.nm = 1
            self.D = np.expand_dims(D, 0)

        # Create space for additive decomposition
        self.add_decomp = None
        self.mult_decomp = None

        # Set F
        if not np.any(F):
            self.F = np.zeros((self.nk, 1))
        else:
            self.F = F

        # Set ν
        if not np.any(ν):
            self.ν = np.zeros((self.nm, 1))
        elif type(ν) == float:
            self.ν = np.asarray([[ν]])
        elif len(ν.shape) == 1:
            self.ν = np.expand_dims(ν, 1)
        else:
            self.ν = ν

        if self.ν.shape[0] != self.D.shape[0]:
            raise ValueError("The dimension of ν is inconsistent with D!")

        # Construct BIG state space representation
        self.lss = self.construct_ss()

    def construct_ss(self):
        """
        This creates the state space representation that can be passed
        into the quantecon LSS class.
        """
        # Pull out useful info
        nx, nk, nm = self.nx, self.nk, self.nm
        A, B, D, F, ν = self.A, self.B, self.D, self.F, self.ν
        if self.add_decomp:
            ν, H, g = self.add_decomp
        else:
            ν, H, g = self.additive_decomp()

        # Auxiliary blocks with 0's and 1's to fill out the lss matrices
        nx0c = np.zeros((nx, 1))
        nx0r = np.zeros(nx)
        nx1 = np.ones(nx)
        nk0 = np.zeros(nk)
        ny0c = np.zeros((nm, 1))
        ny0r = np.zeros(nm)
        ny1m = np.eye(nm)
        ny0m = np.zeros((nm, nm))
        nyx0m = np.zeros_like(D)

        # Build A matrix for LSS
        # Order of states is: [1, t, xt, yt, mt]
        A1 = np.hstack([1, 0, nx0r, ny0r, ny0r])            # Transition for 1
        A2 = np.hstack([1, 1, nx0r, ny0r, ny0r])            # Transition for t
        A3 = np.hstack([nx0c, nx0c, A, nyx0m.T, nyx0m.T])   # Transition for x_{t+1}
        A4 = np.hstack([ν, ny0c, D, ny1m, ny0m])            # Transition for y_{t+1}
        A5 = np.hstack([ny0c, ny0c, nyx0m, ny0m, ny1m])     # Transition for m_{t+1}
        Abar = np.vstack([A1, A2, A3, A4, A5])

        # Build B matrix for LSS
        Bbar = np.vstack([nk0, nk0, B, F, H])

        # Build G matrix for LSS
        # Order of observation is: [xt, yt, mt, st, tt]
        G1 = np.hstack([nx0c, nx0c, np.eye(nx), nyx0m.T, nyx0m.T])    # Selector for x_{t}
        G2 = np.hstack([ny0c, ny0c, nyx0m, ny1m, ny0m])               # Selector for y_{t}
        G3 = np.hstack([ny0c, ny0c, nyx0m, ny0m, ny1m])               # Selector for martingale
        G4 = np.hstack([ny0c, ny0c, -g, ny0m, ny0m])                  # Selector for stationary
        G5 = np.hstack([ny0c, ν, nyx0m, ny0m, ny0m])                  # Selector for trend
        Gbar = np.vstack([G1, G2, G3, G4, G5])

        # Build H matrix for LSS
        Hbar = np.zeros((Gbar.shape[0], nk))

        # Build LSS type
        x0 = np.hstack([1, 0, nx0r, ny0r, ny0r])
        S0 = np.zeros((len(x0), len(x0)))
        lss = qe.lss.LinearStateSpace(Abar, Bbar, Gbar, Hbar, mu_0=x0, Sigma_0=S0)

        return lss

    def additive_decomp(self):
        """
        Return values for the martingale decomposition 
            - ν         : unconditional mean difference in Y
            - H         : coefficient for the (linear) martingale component (κ_a)
            - g         : coefficient for the stationary component g(x)
            - Y_0       : it should be the function of X_0 (for now set it to 0.0)
        """
        I = np.identity(self.nx)
        A_res = la.solve(I - self.A, I)
        g = self.D @ A_res
        H = self.F + self.D @ A_res @ self.B

        return self.ν, H, g

    def multiplicative_decomp(self):
        """
        Return values for the multiplicative decomposition (Example 5.4.4.)
            - ν_tilde  : eigenvalue
            - H        : vector for the Jensen term
        """
        ν, H, g = self.additive_decomp()
        ν_tilde = ν + (.5)*np.expand_dims(np.diag(H @ H.T), 1)

        return ν_tilde, H, g

    def loglikelihood_path(self, x, y):
        A, B, D, F = self.A, self.B, self.D, self.F
        k, T = y.shape
        FF = F @ F.T
        FFinv = la.inv(FF)
        temp = y[:, 1:] - y[:, :-1] - D @ x[:, :-1]
        obs =  temp * FFinv * temp
        obssum = np.cumsum(obs)
        scalar = (np.log(la.det(FF)) + k*np.log(2*np.pi))*np.arange(1, T)

        return -(.5)*(obssum + scalar)

    def loglikelihood(self, x, y):
        llh = self.loglikelihood_path(x, y)

        return llh[-1]


    def plot_additive(self, T, npaths=25, show_trend=True):
        """
        Plots for the additive decomposition

        """
        # Pull out right sizes so we know how to increment
        nx, nk, nm = self.nx, self.nk, self.nm

        # Allocate space (nm is the number of additive functionals - we want npaths for each)
        mpath = np.empty((nm*npaths, T))
        mbounds = np.empty((nm*2, T))
        spath = np.empty((nm*npaths, T))
        sbounds = np.empty((nm*2, T))
        tpath = np.empty((nm*npaths, T))
        ypath = np.empty((nm*npaths, T))

        # Simulate for as long as we wanted
        moment_generator = self.lss.moment_sequence()
        # Pull out population moments
        for t in range (T):
            tmoms = next(moment_generator)
            ymeans = tmoms[1]
            yvar = tmoms[3]

            # Lower and upper bounds - for each additive functional
            for ii in range(nm):
                li, ui = ii*2, (ii+1)*2
                madd_dist = norm(ymeans[nx+nm+ii], np.sqrt(yvar[nx+nm+ii, nx+nm+ii]))
                mbounds[li:ui, t] = madd_dist.ppf([0.01, .99])

                sadd_dist = norm(ymeans[nx+2*nm+ii], np.sqrt(yvar[nx+2*nm+ii, nx+2*nm+ii]))
                sbounds[li:ui, t] = sadd_dist.ppf([0.01, .99])

        # Pull out paths
        for n in range(npaths):
            x, y = self.lss.simulate(T)
            for ii in range(nm):
                ypath[npaths*ii+n, :] = y[nx+ii, :]
                mpath[npaths*ii+n, :] = y[nx+nm + ii, :]
                spath[npaths*ii+n, :] = y[nx+2*nm + ii, :]
                tpath[npaths*ii+n, :] = y[nx+3*nm + ii, :]

        add_figs = []

        for ii in range(nm):
            li, ui = npaths*(ii), npaths*(ii+1)
            LI, UI = 2*(ii), 2*(ii+1)
            add_figs.append(self.plot_given_paths(T, ypath[li:ui,:], mpath[li:ui,:], spath[li:ui,:],
                                                  tpath[li:ui,:], mbounds[LI:UI,:], sbounds[LI:UI,:],
                                                  show_trend=show_trend))

            add_figs[ii].suptitle(f'Additive decomposition of $y_{ii+1}$', fontsize=14)

        return add_figs


    def plot_multiplicative(self, T, npaths=25, show_trend=True):
        """
        Plots for the multiplicative decomposition

        """
        # Pull out right sizes so we know how to increment
        nx, nk, nm = self.nx, self.nk, self.nm
        # Matrices for the multiplicative decomposition
        ν_tilde, H, g = self.multiplicative_decomp()

        # Allocate space (nm is the number of functionals - we want npaths for each)
        mpath_mult = np.empty((nm*npaths, T))
        mbounds_mult = np.empty((nm*2, T))
        spath_mult = np.empty((nm*npaths, T))
        sbounds_mult = np.empty((nm*2, T))
        tpath_mult = np.empty((nm*npaths, T))
        ypath_mult = np.empty((nm*npaths, T))

        # Simulate for as long as we wanted
        moment_generator = self.lss.moment_sequence()
        # Pull out population moments
        for t in range(T):
            tmoms = next(moment_generator)
            ymeans = tmoms[1]
            yvar = tmoms[3]

            # Lower and upper bounds - for each multiplicative functional
            for ii in range(nm):
                li, ui = ii*2, (ii+1)*2
                Mdist = lognorm(np.asscalar(np.sqrt(yvar[nx+nm+ii, nx+nm+ii])), 
                                scale=np.asscalar( np.exp( ymeans[nx+nm+ii]- \
                                                t*(.5)*np.expand_dims(np.diag(H @ H.T),1)[ii])))
                Sdist = lognorm(np.asscalar(np.sqrt(yvar[nx+2*nm+ii, nx+2*nm+ii])),
                                scale = np.asscalar( np.exp(-ymeans[nx+2*nm+ii])))
                mbounds_mult[li:ui, t] = Mdist.ppf([.01, .99])
                sbounds_mult[li:ui, t] = Sdist.ppf([.01, .99])

        # Pull out paths
        for n in range(npaths):
            x, y = self.lss.simulate(T)
            for ii in range(nm):
                ypath_mult[npaths*ii+n, :] = np.exp(y[nx+ii, :])
                mpath_mult[npaths*ii+n, :] = np.exp(y[nx+nm + ii, :] - np.arange(T)*(.5)*np.expand_dims(np.diag(H @ H.T),1)[ii])
                spath_mult[npaths*ii+n, :] = 1/np.exp(-y[nx+2*nm + ii, :])
                tpath_mult[npaths*ii+n, :] = np.exp(y[nx+3*nm + ii, :] + np.arange(T)*(.5)*np.expand_dims(np.diag(H @ H.T),1)[ii])

        mult_figs = []

        for ii in range(nm):
            li, ui = npaths*(ii), npaths*(ii+1)
            LI, UI = 2*(ii), 2*(ii+1)

            mult_figs.append(self.plot_given_paths(T, ypath_mult[li:ui,:], mpath_mult[li:ui,:], 
                                                   spath_mult[li:ui,:], tpath_mult[li:ui,:], 
                                                   mbounds_mult[LI:UI,:], sbounds_mult[LI:UI,:], 1, 
                                                   show_trend=show_trend))
            mult_figs[ii].suptitle(f'Multiplicative decomposition of $y_{ii+1}$', fontsize=14)

        return mult_figs

    def plot_martingales(self, T, npaths=25):

        # Pull out right sizes so we know how to increment
        nx, nk, nm = self.nx, self.nk, self.nm
        # Matrices for the multiplicative decomposition
        ν_tilde, H, g = self.multiplicative_decomp()

        # Allocate space (nm is the number of functionals - we want npaths for each)
        mpath_mult = np.empty((nm*npaths, T))
        mbounds_mult = np.empty((nm*2, T))

        # Simulate for as long as we wanted
        moment_generator = self.lss.moment_sequence()
        # Pull out population moments
        for t in range (T):
            tmoms = next(moment_generator)
            ymeans = tmoms[1]
            yvar = tmoms[3]

            # Lower and upper bounds - for each functional
            for ii in range(nm):
                li, ui = ii*2, (ii+1)*2
                Mdist = lognorm(np.asscalar(np.sqrt(yvar[nx+nm+ii, nx+nm+ii])), 
                                scale=np.asscalar( np.exp( ymeans[nx+nm+ii]- \
                                                t*(.5)*np.expand_dims(np.diag(H @ H.T),1)[ii])))
                mbounds_mult[li:ui, t] = Mdist.ppf([.01, .99])

        # Pull out paths
        for n in range(npaths):
            x, y = self.lss.simulate(T)
            for ii in range(nm):
                mpath_mult[npaths*ii+n, :] = np.exp(y[nx+nm + ii, :] - np.arange(T)*(.5)*np.expand_dims(np.diag(H @ H.T),1)[ii])

        mart_figs = []

        for ii in range(nm):
            li, ui = npaths*(ii), npaths*(ii+1)
            LI, UI = 2*(ii), 2*(ii+1)
            mart_figs.append(self.plot_martingale_paths(T, mpath_mult[li:ui, :],
                                                        mbounds_mult[LI:UI, :],
                                                        horline=1))
            mart_figs[ii].suptitle(f'Martingale components for many paths of $y_{ii+1}$', fontsize=14)

        return mart_figs


    def plot_given_paths(self, T, ypath, mpath, spath, tpath,
                         mbounds, sbounds, horline=0, show_trend=True):

        # Allocate space
        trange = np.arange(T)

        # Create figure
        fig, ax = plt.subplots(2, 2, sharey=True, figsize=(15, 8))

        # Plot all paths together
        ax[0, 0].plot(trange, ypath[0, :], label="$y_t$", color="k")
        ax[0, 0].plot(trange, mpath[0, :], label="$m_t$", color="m")
        ax[0, 0].plot(trange, spath[0, :], label="$s_t$", color="g")
        if show_trend:
            ax[0, 0].plot(trange, tpath[0, :], label="$t_t$", color="r")
        ax[0, 0].axhline(horline, color="k", linestyle="-.")
        ax[0, 0].set_title("One Path of All Variables")
        ax[0, 0].legend(loc="upper left")

        # Plot Martingale Component
        ax[0, 1].plot(trange, mpath[0, :], "m")
        ax[0, 1].plot(trange, mpath.T, alpha=0.45, color="m")
        ub = mbounds[1, :]
        lb = mbounds[0, :]
        ax[0, 1].fill_between(trange, lb, ub, alpha=0.25, color="m")
        ax[0, 1].set_title("Martingale Components for Many Paths")
        ax[0, 1].axhline(horline, color="k", linestyle="-.")

        # Plot Stationary Component
        ax[1, 0].plot(spath[0, :], color="g")
        ax[1, 0].plot(spath.T, alpha=0.25, color="g")
        ub = sbounds[1, :]
        lb = sbounds[0, :]
        ax[1, 0].fill_between(trange, lb, ub, alpha=0.25, color="g")
        ax[1, 0].axhline(horline, color="k", linestyle="-.")
        ax[1, 0].set_title("Stationary Components for Many Paths")

        # Plot Trend Component
        if show_trend:
            ax[1, 1].plot(tpath.T, color="r")
        ax[1, 1].set_title("Trend Components for Many Paths")
        ax[1, 1].axhline(horline, color="k", linestyle="-.")

        return fig

    def plot_martingale_paths(self, T, mpath, mbounds,
                              horline=1, show_trend=False):
        # Allocate space
        trange = np.arange(T)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot Martingale Component
        ub = mbounds[1, :]
        lb = mbounds[0, :]
        ax.fill_between(trange, lb, ub, color="#ffccff")
        ax.axhline(horline, color="k", linestyle="-.")
        ax.plot(trange, mpath.T, linewidth=0.25, color="#4c4c4c")

        return fig
