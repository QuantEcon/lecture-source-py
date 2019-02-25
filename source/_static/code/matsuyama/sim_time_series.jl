function plot_timeseries(n1_0::Real, n2_0::Real,
                         s1::Real=0.5, θ::Real=2.5,
                         δ::Real=0.7, ρ::Real=0.2;
                         ax::PyCall.PyObject=subplots()[2])
    """
    Plot a single time series with initial conditions
    """

    # Create the MSG Model and simulate with initial conditions
    model = MSGSync(s1, θ, δ, ρ)
    n1, n2 = simulate_n(model, n1_0, n2_0, 25)

    ax[:plot](0:24, n1, label=L"$n_1$", lw=2)
    ax[:plot](0:24, n2, label=L"$n_2$", lw=2)

    ax[:legend]()
    ax[:set_ylim](0.15, 0.8)

    return ax
end

# Create figure
fig, ax = subplots(2, 1, figsize=(10, 8))

plot_timeseries(0.15, 0.35, ax=ax[1])
plot_timeseries(0.4, 0.3, ax=ax[2])

ax[1][:set_title]("Not Synchronized")
ax[2][:set_title]("Synchronized")

tight_layout()

show()
