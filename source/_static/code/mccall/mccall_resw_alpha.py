import matplotlib.pyplot as plt

grid_size = 25  
α_vals = np.linspace(0.05, 0.5, grid_size)  
w_bar_vals = np.empty_like(α_vals)

mcm = McCallModel()

fig, ax = plt.subplots(figsize=(10, 6))

for i, α in enumerate(α_vals):
    mcm.α = α
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar

ax.set_xlabel('job separation rate')
ax.set_ylabel('reservation wage')
ax.set_xlim(α_vals.min(), α_vals.max())
txt = r'$\bar w$ as a function of $\alpha$'
ax.plot(α_vals, w_bar_vals, 'b-', lw=2, alpha=0.7, label=txt)
ax.legend(loc='upper right')
ax.grid()

plt.show()
