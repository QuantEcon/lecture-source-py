import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fmin_slsqp
from quantecon import MarkovChain
from scipy.optimize import root


class RecursiveAllocation:

    '''
    Compute the planner's allocation by solving Bellman
    equation.
    '''

    def __init__(self, model, μgrid):

        self.β, self.π, self.G = model.β, model.π, model.G
        self.mc, self.S = MarkovChain(self.π), len(model.π)  # Number of states
        self.Θ, self.model, self.μgrid = model.Θ, model, μgrid

        # Find the first best allocation
        self.solve_time1_bellman()
        self.T.time_0 = True  # Bellman equation now solves time 0 problem


    def solve_time1_bellman(self):
        '''
        Solve the time 1 Bellman equation for calibration model and initial
        grid μgrid0
        '''
        model, μgrid0 = self.model, self.μgrid
        S = len(model.π)

        # First get initial fit
        pp = SequentialAllocation(model)
        c, n, x, V = map(np.vstack, zip(*map(lambda μ: pp.time1_value(μ), μgrid0)))

        Vf, cf, nf, xprimef = {}, {}, {}, {}
        for s in range(2):
            ind = np.argsort(x[:, s])   # Sort x
            # Sort arrays according to x
            c, n, x, V = c[ind], n[ind], x[ind], V[ind]
            cf[s] = UnivariateSpline(x[:, s], c[:, s])
            nf[s] = UnivariateSpline(x[:, s], n[:, s])
            Vf[s] = UnivariateSpline(x[:, s], V[:, s])
            for sprime in range(S):
                xprimef[s, sprime] = UnivariateSpline(x[:, s], x[:, s])
        policies = [cf, nf, xprimef]

        # Create xgrid
        xbar = [x.min(0).max(), x.max(0).min()]
        xgrid = np.linspace(xbar[0], xbar[1], len(μgrid0))
        self.xgrid = xgrid

        # Now iterate on bellman equation
        T = BellmanEquation(model, xgrid, policies)
        diff = 1
        while diff > 1e-7:
            PF = T(Vf)
            Vfnew, policies = self.fit_policy_function(PF)
            diff = 0
            for s in range(S):
                diff = max(diff, np.abs(
                    (Vf[s](xgrid) - Vfnew[s](xgrid)) / Vf[s](xgrid)).max())
            Vf = Vfnew

        # Store value function policies and Bellman Equations
        self.Vf = Vf
        self.policies = policies
        self.T = T


    def fit_policy_function(self, PF):
        '''
        Fits the policy functions PF using the points xgrid using
        UnivariateSpline
        '''
        xgrid, S = self.xgrid, self.S

        Vf, cf, nf, xprimef = {}, {}, {}, {}
        for s in range(S):
            PFvec = np.vstack(tuple(map(lambda x: PF(x, s), xgrid)))
            Vf[s] = UnivariateSpline(xgrid, PFvec[:, 0], s=0)
            cf[s] = UnivariateSpline(xgrid, PFvec[:, 1], s=0, k=1)
            nf[s] = UnivariateSpline(xgrid, PFvec[:, 2], s=0, k=1)
            for sprime in range(S):
                xprimef[s, sprime] = UnivariateSpline(
                    xgrid, PFvec[:, 3 + sprime], s=0, k=1)

        return Vf, [cf, nf, xprimef]


    def Τ(self, c, n):
        '''
        Computes Τ given c, n
        '''
        model = self.model
        Uc, Un = model.Uc(c, n), model.Un(c, n)

        return 1 + Un / (self.Θ * Uc)


    def time0_allocation(self, B_, s0):
        '''
        Finds the optimal allocation given initial government debt B_ and
        state s_0
        '''
        PF = self.T(self.Vf)
        z0 = PF(B_, s0)
        c0, n0, xprime0 = z0[1], z0[2], z0[3:]
        return c0, n0, xprime0


    def simulate(self, B_, s_0, T, sHist=None):
        '''
        Simulates Ramsey plan for T periods
        '''
        model, π = self.model, self.π
        Uc = model.Uc
        cf, nf, xprimef = self.policies

        if sHist is None:
            sHist = self.mc.simulate(T, s_0)

        cHist, nHist, Bhist, ΤHist, μHist = np.zeros((5, T))
        RHist = np.zeros(T - 1)

        # Time 0
        cHist[0], nHist[0], xprime = self.time0_allocation(B_, s_0)
        ΤHist[0] = self.Τ(cHist[0], nHist[0])[s_0]
        Bhist[0] = B_
        μHist[0] = 0

        # Time 1 onward
        for t in range(1, T):
            s, x = sHist[t], xprime[sHist[t]]
            c, n, xprime = np.empty(self.S), nf[s](x), np.empty(self.S)
            for shat in range(self.S):
                c[shat] = cf[shat](x)
            for sprime in range(self.S):
                xprime[sprime] = xprimef[s, sprime](x)

            Τ = self.Τ(c, n)[s]
            u_c = Uc(c, n)
            Eu_c = π[sHist[t - 1]] @ u_c
            μHist[t] = self.Vf[s](x, 1)

            RHist[t - 1] = Uc(cHist[t - 1], nHist[t - 1]) / (self.β * Eu_c)

            cHist[t], nHist[t], Bhist[t], ΤHist[t] = c[s], n, x / u_c[s], Τ

        return np.array([cHist, nHist, Bhist, ΤHist, sHist, μHist, RHist])


class BellmanEquation:

    '''
    Bellman equation for the continuation of the Lucas-Stokey Problem
    '''

    def __init__(self, model, xgrid, policies0):

        self.β, self.π, self.G = model.β, model.π, model.G
        self.S = len(model.π)  # Number of states
        self.Θ, self.model = model.Θ, model

        self.xbar = [min(xgrid), max(xgrid)]
        self.time_0 = False

        self.z0 = {}
        cf, nf, xprimef = policies0
        for s in range(self.S):
            for x in xgrid:
                xprime0 = np.empty(self.S)
                for sprime in range(self.S):
                    xprime0[sprime] = xprimef[s, sprime](x)
                self.z0[x, s] = np.hstack([cf[s](x), nf[s](x), xprime0])

        self.find_first_best()


    def find_first_best(self):
        '''
        Find the first best allocation
        '''
        model = self.model
        S, Θ, Uc, Un, G = self.S, self.Θ, model.Uc, model.Un, self.G

        def res(z):
            c = z[:S]
            n = z[S:]
            return np.hstack([Θ * Uc(c, n) + Un(c, n), Θ * n - c - G])

        res = root(res, 0.5 * np.ones(2 * S))
        if not res.success:
            raise Exception('Could not find first best')

        self.cFB = res.x[:S]
        self.nFB = res.x[S:]
        IFB = Uc(self.cFB, self.nFB) * self.cFB + Un(self.cFB, self.nFB) * self.nFB
        self.xFB = np.linalg.solve(np.eye(S) - self.β * self.π, IFB)
        self.zFB = {}

        for s in range(S):
            self.zFB[s] = np.hstack([self.cFB[s], self.nFB[s], self.xFB])


    def __call__(self, Vf):
        '''
        Given continuation value function, next period return value function,
        this period return T(V) and optimal policies
        '''
        if not self.time_0:
            def PF(x, s): return self.get_policies_time1(x, s, Vf)
        else:
            def PF(B_, s0): return self.get_policies_time0(B_, s0, Vf)
        return PF


    def get_policies_time1(self, x, s, Vf):
        '''
        Finds the optimal policies
        '''
        model, β, Θ, = self.model, self.β, self.Θ,
        G, S, π = self.G, self.S, self.π
        U, Uc, Un = model.U, model.Uc, model.Un

        def objf(z):
            c, n, xprime = z[0], z[1], z[2:]
            Vprime = np.empty(S)
            for sprime in range(S):
                Vprime[sprime] = Vf[sprime](xprime[sprime])

            return -(U(c, n) + β * π[s] @ Vprime)

        def cons(z):
            c, n, xprime = z[0], z[1], z[2:]
            return np.hstack([x - Uc(c, n) * c - Un(c, n) * n - β * π[s]
                              @ xprime,
                              (Θ * n - c - G)[s]])

        out, fx, _, imode, smode = fmin_slsqp(objf,
                                              self.z0[x, s],
                                              f_eqcons=cons,
                                              bounds=[(0, 100), (0, 100)]
                                              + [self.xbar] * S,
                                              full_output=True,
                                              iprint=0,
                                              acc=1e-10)

        if imode > 0:
            raise Exception(smode)

        self.z0[x, s] = out
        return np.hstack([-fx, out])


    def get_policies_time0(self, B_, s0, Vf):
        '''
        Finds the optimal policies
        '''
        model, β, Θ, = self.model, self.β, self.Θ,
        G, S, π = self.G, self.S, self.π
        U, Uc, Un = model.U, model.Uc, model.Un

        def objf(z):
            c, n, xprime = z[0], z[1], z[2:]
            Vprime = np.empty(S)
            for sprime in range(S):
                Vprime[sprime] = Vf[sprime](xprime[sprime])

            return -(U(c, n) + β * π[s0] @ Vprime)

        def cons(z):
            c, n, xprime = z[0], z[1], z[2:]
            return np.hstack([-Uc(c, n) * (c - B_) - Un(c, n) * n - β * π[s0]
                              @ xprime,
                              (Θ * n - c - G)[s0]])

        out, fx, _, imode, smode = fmin_slsqp(objf, self.zFB[s0], f_eqcons=cons,
                                              bounds=[(0, 100), (0, 100)]
                                              + [self.xbar] * S,
                                              full_output=True, iprint=0,
                                              acc=1e-10)

        if imode > 0:
            raise Exception(smode)

        return np.hstack([-fx, out])
