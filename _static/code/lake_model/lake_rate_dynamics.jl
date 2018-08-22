lm = LakeModel()
e_0 = 0.92     # Initial employment rate
u_0 = 1 - e_0  # Initial unemployment rate
T = 50         # Simulation length

xbar = rate_steady_state(lm)
x_0 = [u_0; e_0]
x_path = simulate_rate_path(lm, x_0, T)

titles = ["Unmployment rate" "Employment rate"]

fig, axes = subplots(2, 1, figsize=(10, 8))

for (i, ax) in enumerate(axes)
    ax[:plot](1:T, x_path[i, :], c="blue", lw=2, alpha=0.5)
    ax[:hlines](xbar[i], 0, T, "r", "--")
    ax[:set](title=titles[i])
    ax[:grid]("on")
end