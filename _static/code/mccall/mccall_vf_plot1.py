import matplotlib.pyplot as plt

mcm = McCallModel()
e, h = solve_mccall_model(mcm)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(mcm.w_vals, e, 'b-', lw=2, alpha=0.7, label='$e$')
ax.plot(mcm.w_vals, [h]*len(mcm.w_vals), 'g-', lw=2, alpha=0.7, label='$h$')
ax.set_xlim(min(mcm.w_vals), max(mcm.w_vals))
ax.legend()
ax.grid()

plt.show()
