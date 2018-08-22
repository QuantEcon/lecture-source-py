grid_size = 25  
β_vals = np.linspace(0.8, 0.99, grid_size)  
w_bar_vals = np.empty_like(β_vals)

mcm = McCallModel()

fig, ax = plt.subplots(figsize=(10, 6))

for i, β in enumerate(β_vals):
    mcm.β = β
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar

ax.set_xlabel('discount factor')
ax.set_ylabel('reservation wage')
ax.set_xlim(β_vals.min(), β_vals.max())
txt = r'$\bar w$ as a function of $\beta$'
ax.plot(β_vals, w_bar_vals, 'b-', lw=2, alpha=0.7, label=txt)
ax.legend(loc='upper left')
ax.grid()

plt.show()
