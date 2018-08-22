grid_size = 25  
γ_vals = linspace(0.05, 0.95, grid_size)  
w_bar_vals = similar(γ_vals)

mcm = McCallModel()

for (i, γ) in enumerate(γ_vals)
    mcm.γ = γ
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar
end

plot(γ_vals, 
     w_bar_vals, 
     lw=2, 
     α=0.7, 
     xlabel="job offer rate",
     ylabel="reservation wage",
     label=L"$\bar w$ as a function of $γ$")