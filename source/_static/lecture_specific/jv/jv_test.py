import matplotlib.pyplot as plt
from quantecon import compute_fixed_point

# === solve for optimal policy === #
wp = JvWorker(grid_size=25)
v_init = wp.x_grid * 0.5
V = compute_fixed_point(wp.bellman_operator, v_init, max_iter=40)
s_policy, ϕ_policy = wp.bellman_operator(V, return_policies=True)

# === plot policies === #

plots = [ϕ_policy, s_policy, V]
titles = ["ϕ policy", "s policy", "value function"]

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

for ax, plot, title in zip(axes, plots, titles):
    ax.plot(wp.x_grid, plot)
    ax.set(title=title)
    ax.grid()

axes[-1].set_xlabel("x")
plt.show()
