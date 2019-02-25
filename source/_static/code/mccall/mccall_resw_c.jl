grid_size = 25  
c_vals = linspace(2, 12, grid_size)  
w_bar_vals = similar(c_vals)

mcm = McCallModel()

for (i, c) in enumerate(c_vals)
    mcm.c = c
    w_bar = compute_reservation_wage(mcm)
    w_bar_vals[i] = w_bar
end

plot(c_vals, 
     w_bar_vals, 
     lw=2, 
     α=0.7, 
     xlabel="unemployment compensation",
     ylabel="reservation wage",
     label=L"$\bar w$ as a function of $c$")

