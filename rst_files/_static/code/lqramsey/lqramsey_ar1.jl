# == Parameters == #
β = 1 / 1.05
ρ, mg = .7, .35
A = eye(2)
A = [ρ mg*(1 - ρ); 0.0 1.0]
C = [sqrt(1 - ρ^2) * mg / 10 0.0; 0 0]
Sg = [1.0 0.0]
Sd = [0.0 0.0]
Sb = [0 2.135]
Ss = [0.0 0.0]
discrete = false
proc = ContStochProcess(A, C)

econ = Economy(β, Sg, Sd, Sb, Ss, discrete, proc)
T = 50
path = compute_paths(econ, T)

gen_fig_1(path)