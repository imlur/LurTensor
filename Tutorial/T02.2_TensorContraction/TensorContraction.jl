include("../../Tensor/LurTensor.jl")
using Random

# Initialization
d_a, d_b, d_c, d_d, d_m = 5, 6, 7, 8, 9 # d_alpha, d_beta, d_gamma, d_delta, d_mu

Random.seed!(720)
A = rand(d_c, d_d)
B = rand(d_a, d_m, d_c)
C = rand(d_b, d_m, d_d)


# Method 1. Using matrix multiplication
B1 = permutedims(B, [1 3 2])
B1 = reshape(B1, d_a*d_c, d_m)
C1 = permutedims(C, [2 1 3])
C1 = reshape(C1, d_m, d_b*d_d)

BC = B1 * C1
BC = reshape(BC, d_a, d_c, d_b, d_d)

A1 = reshape(A, d_c*d_d)
BC1 = permutedims(BC, [1 3 2 4])
BC1 = reshape(BC1, d_a*d_b, d_c*d_d)

ABC = BC1 * A1
ABC = reshape(ABC, d_a, d_b)

println("From multiplication :")
show(stdout, "text/plain", ABC)
println("\n\n")

# Method 2. Using LurTensor. (Automated reshaping and permutating dimensions)
Ai = LurTensor(A, "C", "D")
Bi = LurTensor(B, "A", "M", "C")
Ci = LurTensor(C, "B", "M", "D")

# Multiplication of LurTensors means contraction of common indices
# Contract B and C
BCi = Bi * Ci
# Contract BC and A
ABCi = Ai * BCi

println("From LurTensors :")
println(ABCi)
println("\n\n")

# Time comparison. Matrix multiplication vs. for loops

# Matrix multiplication
start_time = time()
B1 = permutedims(B, [1 3 2])
B1 = reshape(B1, d_a*d_c, d_m)
C1 = permutedims(C, [2 1 3])
C1 = reshape(C1, d_m, d_b*d_d)

BC = B1 * C1
BC = reshape(BC, d_a, d_c, d_b, d_d)
end_time = time()

println("$(round(end_time-start_time, digits=4))s from matrix multiplication method\n\n")

# for loops
start_time = time()
BC = zeros(d_a, d_c, d_b, d_d)
for ai=1:d_a
	for ci=1:d_c
		for bi=1:d_b
			for di=1:d_d
				for mi=1:d_m
					BC[ai, ci, bi, di] += B[ai, mi, ci] * C[bi, mi, di]
				end
			end
		end
	end
end

end_time = time()
println("$(round(end_time-start_time, digits=4))s from for-loop method")
