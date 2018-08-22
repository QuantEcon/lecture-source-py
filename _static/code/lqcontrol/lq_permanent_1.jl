using QuantEcon
using Plots
using Plots.PlotMeasures
pyplot()

# == Model parameters == #
r = 0.05
β = 1/(1 + r)
T = 45
c_bar = 2.0
σ = 0.25
μ = 1.0
q = 1e6

# == Formulate as an LQ problem == #
Q = 1.0
R = zeros(2, 2)
Rf = zeros(2, 2); Rf[1, 1] = q
A = [1.0+r -c_bar+μ;
     0.0   1.0     ]
B = [-1.0; 0.0]
C = [σ; 0.0]

# == Compute solutions and simulate == #
lq = LQ(Q, R, A, B, C; bet=β, capT=T, rf=Rf)
x0 = [0.0; 1.0]
xp, up, wp = compute_sequence(lq, x0)

# == Convert back to assets, consumption and income == #
assets = vec(xp[1, :])               # a_t
c = vec(up + c_bar)                  # c_t
income = vec(σ * wp[1, 2:end] + μ)   # y_t

# == Plot results == #
p=plot(Vector[assets, c, zeros(T + 1), income, cumsum(income - μ)],
  lab=["assets" "consumption" "" "non-financial income" "cumulative unanticipated income"],
  color=[:blue :green :black :orange :red],
  xaxis=("Time"), layout=(2, 1),
  bottom_margin = 20mm, size=(600, 600))


