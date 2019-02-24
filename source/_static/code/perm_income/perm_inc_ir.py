r = 0.05
β = 1 / (1 + r)
T = 20  # Time horizon
S = 5   # Impulse date
σ1 = σ2 = 0.15


def time_path(permanent=False):
    "Time path of consumption and debt given shock sequence"
    w1 = np.zeros(T+1)
    w2 = np.zeros(T+1)
    b = np.zeros(T+1)
    c = np.zeros(T+1)
    if permanent:
        w1[S+1] = 1.0
    else:
        w2[S+1] = 1.0
    for t in range(1, T):
        b[t+1] = b[t] - σ2 * w2[t]
        c[t+1] = c[t] + σ1 * w1[t+1] + (1 - β) * σ2 * w2[t+1]
    return b, c


fig, axes = plt.subplots(2, 1, figsize=(10, 8))
p_args = {'lw': 2, 'alpha': 0.7}
titles = ['transitory', 'permanent']

L = 0.175

for ax, truefalse, title in zip(axes, (True, False), titles):
    b, c = time_path(permanent=truefalse)
    ax.set_title(f'Impulse reponse: {title} income shock')
    ax.plot(list(range(T+1)), c, 'g-', label="consumption", **p_args)
    ax.plot(list(range(T+1)), b, 'b-', label="debt", **p_args)
    ax.plot((S, S), (-L, L), 'k-', lw=0.5)
    ax.grid(alpha=0.5)
    ax.set(xlabel=r'Time', ylim=(-L, L))

axes[0].legend(loc='lower right')

plt.tight_layout()
plt.show()