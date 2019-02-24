grid_size = 25  
β_vals = linspace(0.8, 0.99, grid_size)  
w_bar_vals = similar(β_vals)

mcm = McCallModel()

for (i, β) in enumerate(β_vals)
    mcm.β = β
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar
end

plot(β_vals, 
     w_bar_vals, 
     lw=2, 
     α=0.7, 
     xlabel="discount rate",
     ylabel="reservation wage",
     label=L"$\bar w$ as a function of $\β$")

