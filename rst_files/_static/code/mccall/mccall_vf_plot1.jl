using Plots, LaTeXStrings
pyplot()


mcm = McCallModel()
V, U = solve_mccall_model(mcm)
U_vec = U .* ones(length(mcm.w_vec))

plot(mcm.w_vec, 
     [V U_vec],
     lw=2, 
     α=0.7, 
     label=[L"$V$" L"$U$"])

