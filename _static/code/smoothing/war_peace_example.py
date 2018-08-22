# Parameters

β = .96
y = [1, 2]
b0 = 0
P = np.asarray([[.8, .2],
                [.4, .6]])

cp = ConsumptionProblem(β, y, b0, P)
Q = β * P
N_simul = 150

c_bar, b1, b2 = consumption_complete(cp)
debt_complete = np.asarray([b1, b2])

print(f"P \n {P}")
print(f"Q \n {Q}")
print(f"Govt expenditures in peace and war = {y}")
print(f"Constant tax collections = {c_bar}")
print(f"Govt assets in two states = {debt_complete}")

msg = """
Now let's check the government's budget constraint in peace and war.
Our assumptions imply that the government always purchases 0 units of the
Arrow peace security.
"""
print(msg)

AS1 = Q[0, 1] * b2
print(f"Spending on Arrow war security in peace = {AS1}")
AS2 = Q[1, 1] * b2
print(f"Spending on Arrow war security in war = {AS2}")

print("\n")
print("Government tax collections plus asset levels in peace and war")
TB1 = c_bar + b1
print(f"T+b in peace = {TB1}")
TB2 = c_bar + b2
print(f"T+b in war = {TB2}")

print("\n")
print("Total government spending in peace and war")
G1 = y[0] + AS1
G2 = y[1] + AS2
print(f"Peace = {G1}")
print(f"War = {G2}")

print("\n")
print("Let's see ex post and ex ante returns on Arrow securities")

Π = np.reciprocal(Q)
exret = Π
print(f"Ex post returns to purchase of Arrow securities = {exret}")
exant = Π * P
print(f"Ex ante returns to purchase of Arrow securities {exant}")