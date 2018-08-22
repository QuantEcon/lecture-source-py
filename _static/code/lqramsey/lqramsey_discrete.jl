# Parameters
β = 1 / 1.05
P = [0.8 0.2 0.0
     0.0 0.5 0.5
     0.0 0.0 1.0]

# Possible states of the world
# Each column is a state of the world. The rows are [g d b s 1]
x_vals = [0.5 0.5 0.25;
          0.0 0.0  0.0;
          2.2 2.2  2.2;
          0.0 0.0  0.0;
          1.0 1.0  1.0]
Sg = [1.0 0.0 0.0 0.0 0.0]
Sd = [0.0 1.0 0.0 0.0 0.0]
Sb = [0.0 0.0 1.0 0.0 0.0]
Ss = [0.0 0.0 0.0 1.0 0.0]
discrete = true
proc = DiscreteStochProcess(P, x_vals)

econ = Economy(β, Sg, Sd, Sb, Ss, discrete, proc)
T = 15
path = compute_paths(econ, T)

gen_fig_1(path)