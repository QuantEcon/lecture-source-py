import scipy.interpolate as interp
import quantecon as qe

def expect_loss_choose_0(p, L0):
    "For a given probability return expected loss of choosing model 0"
    return (1 - p) * L0

def expect_loss_choose_1(p, L1):
    "For a given probability return expected loss of choosing model 1"
    return p * L1

def EJ(p, f0, f1, J):
    """
    Evaluates the expectation of our value function J. To do this, we
    need the current probability that model 0 is correct (p), the
    distributions (f0, f1), and the function J.
    """
    # Get the current distribution we believe (p*f0 + (1-p)*f1)
    curr_dist = p * f0 + (1 - p) * f1
    
    # Get tomorrow's expected distribution through Bayes law
    tp1_dist = np.clip((p * f0) / (p * f0 + (1 - p) * f1), 0, 1)
    
    # Evaluate the expectation
    EJ = curr_dist @ J(tp1_dist)
    
    return EJ

def expect_loss_cont(p, c, f0, f1, J):
    return c + EJ(p, f0, f1, J)


def bellman_operator(pgrid, c, f0, f1, L0, L1, J):
    """
    Evaluates the value function for a given continuation value
    function; that is, evaluates

        J(p) = min((1 - p) L0, p L1, c + E J(p'))

    Uses linear interpolation between points.
    """
    m = np.size(pgrid)
    assert m == np.size(J)
    
    J_out = np.zeros(m)
    J_interp = interp.UnivariateSpline(pgrid, J, k=1, ext=0)

    for (p_ind, p) in enumerate(pgrid):
        # Payoff of choosing model 0
        p_c_0 = expect_loss_choose_0(p, L0)
        p_c_1 = expect_loss_choose_1(p, L1)
        p_con = expect_loss_cont(p, c, f0, f1, J_interp)
        
        J_out[p_ind] = min(p_c_0, p_c_1, p_con)

    return J_out


#  == Now run at given parameters == #

#  First set up distributions 
p_m1 = np.linspace(0, 1, 50)
f0 = np.clip(st.beta.pdf(p_m1, a=1, b=1), 1e-8, np.inf)
f0 = f0 / np.sum(f0)
f1 = np.clip(st.beta.pdf(p_m1, a=9, b=9), 1e-8, np.inf)
f1 = f1 / np.sum(f1)

# Build a grid
pg = np.linspace(0, 1, 251)
# Turn the Bellman operator into a function with one argument
bell_op = lambda vf: bellman_operator(pg, 0.5, f0, f1, 5.0, 5.0, vf)
# Pass it to qe's built in iteration routine
J = qe.compute_fixed_point(bell_op, 
                            np.zeros(pg.size),  # Initial guess
                            error_tol=1e-6, 
                            verbose=True, 
                            print_skip=5)

