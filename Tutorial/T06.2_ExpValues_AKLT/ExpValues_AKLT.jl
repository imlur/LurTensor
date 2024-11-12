include("../../Tensor/LurTensor.jl")

function check_normalization(T::LurTensor)
	Tl = conj(prime(T; tag="L"))
	Tr = conj(prime(T; tag="R"))

	println("Checking right-normalization")
	println(T * Tl)

	println("Checking left-normalization")
	println(T * Tr)
end

# Generate AKLT state 
AKLT = zeros(2, 2, 3)

# Local spin S_z = +1
AKLT[1, 2, 1] = sqrt(2/3)
# Local spin S_z = 0
AKLT[1, 1, 2] = -1 / sqrt(3)
AKLT[2, 2, 2] = 1 / sqrt(3)
# Local spin S_z = -1
AKLT[2, 1, 3] = -sqrt(2/3)

# Normalization of AKLT Tensor
T = LurTensor(AKLT, "L", "R", "S") # Right, Left, Site
T1 = T[1:1, :, :]; T2 = T[:, 1:1, :]

println("\nFor T,")
check_normalization(T)
println("\nFor T1,")
check_normalization(T1)
println("\nFor T2,")
check_normalization(T2)


let 
	L = 50;
	M = Vector{LurTensor}(undef, L)
	site_tags = ["site,$(i)" for i=1:L]
	bond_tags = ["ld", ["$(i)~$(i+1)" for i=1:L-1]..., "rd"]

	for i=1:L
		M[i] = LurTensor(AKLT, bond_tags[i], bond_tags[i+1], site_tags[i])
	end
	
	# Project the space of the left leg of M[1] and the space of the
	# right leg of M[end] onto the subspaces of size 1 for each leg
	M[1] = M[1][1:1, :, :]
	M[end] = M[end][:, 1:1, :]
	M, s, _ = canonform(M, L, "ld", "rd")
	println(s[1]) # 1 / sqrt(2), not normalized

	n = 10
	S = getlocalspace("Spin", 1, 'S', (site_tags[n], 1), (site_tags[n], 0), "_")
	# rank 2 tensor with Index objects ("site,10", 1) and ("site,10", 0)
	Sz = S[:, :, 2] 
	
	T = LurTensor(reshape([1], 1, 1), ["ld", "ld"], [0, 1])
	for i=1:L
		if i == n
			T = updateleft(T, M[i]', Sz, M[i])
		else
			T = updateleft(T, M[i]', (site_tags[i], 1), M[i], (site_tags[i], 0))
		end
	end
	println("\n\nMagnetization at site $(n) is")
	println(value(T))
end
