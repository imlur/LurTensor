include("../../Tensor/LurTensor.jl")
using Random

# Only LurTensor version in this solution
# Initialization
d_a, d_b, d_c, d_d, d_m = 5, 6, 7, 8, 9 # d_alpha, d_beta, d_gamma, d_delta, d_mu

Random.seed!(720)
A = rand(d_c, d_d)
B = rand(d_a, d_m, d_c)
C = rand(d_b, d_m, d_d)

Ai = LurTensor(A, "C", "D")
Bi = LurTensor(B, "A", "M", "C")
Ci = LurTensor(C, "B", "M", "D")

# Contract A and C
ACi = Ai * Ci
# Contract AC and B
ABCi = Bi * ACi
show(stdout, ABCi; showarr=true)
println("\n")
