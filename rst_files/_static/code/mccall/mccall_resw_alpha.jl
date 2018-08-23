grid_size = 25  
α_vals = linspace(0.05, 0.5, grid_size)  
w_bar_vals = similar(α_vals)

mcm = McCallModel()

for (i, α) in enumerate(α_vals)
    mcm.α = α
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar
end

plot(α_vals, 
     w_bar_vals, 
     lw=2, 
     α=0.7, 
     xlabel="job separation rate",
     ylabel="reservation wage",
     label=L"$\bar w$ as a function of $\α$")

