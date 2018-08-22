using QuantEcon

srand(42)
lm = LakeModel(d=0.0, b=0.0)
T = 5000                        # Simulation length

α, λ = lm.α, lm.λ
P = [(1 - λ)     λ; 
     α       (1 - α)]

mc = MarkovChain(P, [0; 1])     # 0=unemployed, 1=employed
xbar = rate_steady_state(lm)

s_path = simulate(mc, T; init=2)
s_bar_e = cumsum(s_path) ./ (1:T)
s_bar_u = 1 - s_bar_e
s_bars = [s_bar_u s_bar_e]

titles = ["Percent of time unemployed" "Percent of time employed"]

fig, axes = subplots(2, 1, figsize=(10, 8))

for (i, ax) in enumerate(axes)
    ax[:plot](1:T, s_bars[:, i], c="blue", lw=2, alpha=0.5)
    ax[:hlines](xbar[i], 0, T, "r", "--")
    ax[:set](title=titles[i])
    ax[:grid]("on")
end
