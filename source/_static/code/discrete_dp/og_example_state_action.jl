using QuantEcon 

B = 10
M = 5
α = 0.5
β = 0.9
u(c) = c^α
n = B + M + 1
m = M + 1

s_indices = Int64[]
a_indices = Int64[]
Q = Array{Float64}(0, n)
R = Float64[]

b = 1.0 / (B + 1)

for s in 0:(M + B)
    for a in 0:min(M, s)
        s_indices = [s_indices; s + 1]
        a_indices = [a_indices; a + 1]
        q = zeros(Float64, 1, n)
        q[(a + 1):((a + B) + 1)] = b 
        Q = [Q; q]
        R = [R; u(s-a)]
    end
end

ddp = DiscreteDP(R, Q, β, s_indices, a_indices);
results = solve(ddp, PFI)
