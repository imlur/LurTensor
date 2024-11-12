include("../../Tensor/LurTensor.jl")

sz = [2 3 2 3 4]
T = reshape((1:prod(sz)), sz...)
T = T / norm(T)
# rank of the tensor
rank = length(sz)

# Get entanglement entropy from singular values
function ee_from_svals(svals)
	return -sum([s==0 ? 0 : s^2*log2(s^2) for s in svals])
end

# using matrix manipulation
println("From matrix manipulation, ")

# (i) A = {1, 2}, B = {3, 4, 5}
Ta = reshape(T, sz[1]*sz[2], sz[3]*sz[4]*sz[5])
svals = svdvals(Ta)
println("Entanglement entropy when A = {1, 2} is $(ee_from_svals(svals))")


# (ii) A = {1, 3}, B = {2, 4, 5}
Tb = permutedims(T, [1, 3, 2, 4, 5])
Tb = reshape(Tb, sz[1]*sz[3], sz[2]*sz[4]*sz[5])
svals = svdvals(Tb)
println("Entanglement entropy when A = {1, 3} is $(ee_from_svals(svals))")


# (iii) A = {1, 5}, B = {2, 3, 4}
Tc = permutedims(T, [1, 5, 2, 3, 4])
Tc = reshape(Tc, sz[1]*sz[5], sz[2]*sz[3]*sz[4])
svals = svdvals(Tc)
println("Entanglement entropy when A = {1, 5} is $(ee_from_svals(svals))")

println("\n\nFrom LurTensors, ")

# LurTensors version
tagvec = ["site$(i)" for i=1:5]
# left dummy leg / right dummy leg
T_withdummy = reshape(T, (1, size(T)..., 1))
Ti = LurTensor(T_withdummy, "ldm", tagvec..., "rdm")

# (i) A = {1, 2}, B = {3, 4, 5}
svals, _ = svdvals(Ti, "ldm", tagvec[1], tagvec[2])
println("Entanglement entropy when A = {1, 2} is $(ee_from_svals(svals))")

# (ii) A = {1, 3}, B = {2, 4, 5}
svals, _ = svdvals(Ti, "ldm", tagvec[1], tagvec[3])
println("Entanglement entropy when A = {1, 3} is $(ee_from_svals(svals))")

# (iii) A = {1, 5}, B = {2, 3, 4}
svals, _ = svdvals(Ti, "ldm", tagvec[1], tagvec[5])
println("Entanglement entropy when A = {1, 5} is $(ee_from_svals(svals))")
