using Plots
pyplot()
using QuantEcon
using LaTeXStrings

phi_1, phi_2, phi_3, phi_4 = 0.5, -0.2, 0, 0.5
sigma = 0.1

A = [phi_1 phi_2 phi_3 phi_4
     1.0 0.0 0.0 0.0
     0.0 1.0 0.0 0.0
     0.0 0.0 1.0 0.0]

C = [sigma 0.0 0.0 0.0]'
G = [1.0 0.0 0.0 0.0]

T = 30
ar = LSS(A, C, G, mu_0=ones(4))

ymin, ymax = -0.8, 1.25

ys = []
yTs = []

for i=1:20
    x, y = simulate(ar, T+15)
    y = squeeze(y, 1)
    push!(ys, y)
    push!(yTs, y[T])
end
p1 = plot(ys, linewidth=1, alpha=0.5)
scatter!(T*ones(1, T), yTs, color=:black)
plot!(ylims=(ymin, ymax), xticks=[], ylabel=L"$y_t$")
vline!([T], color=:black, legend=:none)
annotate!(T+1, -0.8, L"$T$")

p2 = histogram(yTs, bins=16, normed=true, orientation=:h, alpha=0.5)
plot!(ylims=(ymin, ymax))
plot(p1, p2, layout=2, legend=:none)

