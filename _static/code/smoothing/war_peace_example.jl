# Parameters

β = .96
y = [1.0, 2.0]
b0 = 0.0
P = [0.8 0.2;
     0.4 0.6]

cp = ConsumptionProblem(β, y, b0, P)
Q = β * P
N_simul = 150

c_bar, b1, b2 = consumption_complete(cp)
debt_complete = [b1, b2]

println("P = $P")
println("Q = $Q")
println("Govt expenditures in peace and war = $y")
println("Constant tax collections = $c_bar")
println("Govt assets in two states = $debt_complete")

msg = """
Now let's check the government's budget constraint in peace and war.
Our assumptions imply that the government always purchases 0 units of the
Arrow peace security.
"""
println(msg)

AS1 = Q[1, 2] * b2
println("Spending on Arrow war security in peace = $AS1")
AS2 = Q[2, 2] * b2
println("Spending on Arrow war security in war = $AS2")

println("\n")
println("Government tax collections plus asset levels in peace and war")
TB1 = c_bar + b1
println("T+b in peace = $TB1")
TB2 = c_bar + b2
println("T+b in war = $TB2")


println("\n")
println("Total government spending in peace and war")
G1= y[1] + AS1
G2 = y[2] + AS2
println("total govt spending in peace = $G1")
println("total govt spending in war = $G2")


println("\n")
println("Let's see ex post and ex ante returns on Arrow securities")

Π = 1 ./ Q    # reciprocal(Q)
exret = Π
println("Ex post returns to purchase of Arrow securities = $exret")
exant = Π .* P
println("Ex ante returns to purchase of Arrow securities = $exant")