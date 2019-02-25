using PyPlot

AF = A - B1 * F1 - B2 * F2
n = 20
x = Array{Float64}(3, n)
x[:, 1] = [1 1 1]
for t in 1:n-1
    x[:, t+1] = AF * x[:, t]
end
q1 = x[2, :]
q2 = x[3, :]
q = q1 + q2         # Total output, MPE
p = a0 - a1 * q     # Price, MPE

fig, ax = subplots(figsize=(9, 5.8))
ax[:plot](q, "b-", lw=2, alpha=0.75, label="total output")
ax[:plot](p, "g-", lw=2, alpha=0.75, label="price")
ax[:set_title]("Output and prices, duopoly MPE")
ax[:legend](frameon=false)
