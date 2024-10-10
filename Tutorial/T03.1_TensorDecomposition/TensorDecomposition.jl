using LinearAlgebra
include("../../Tensor/LurTensor.jl")


sz = [2 3 2 3 4]
T = reshape((1:prod(sz)), sz...)
T = T / norm(T)
# rank of the tensor
rank = length(sz)

# Matrix manipulation version
let 
	# Decomposition results
	Tensors = Vector{Array{Float64}}(undef, rank)
	R = T; szl = 1

	for i=1:rank-1
		R = reshape(R, szl*sz[i], prod(sz[i+1:end]))
		F = qr(R); Q, R = Matrix(F.Q), F.R
		new_szl = div(length(Q), szl*sz[i])
		Q = reshape(Q, szl, sz[i], new_szl)
		Tensors[i] = permutedims(Q, [1, 3, 2])
		R = reshape(R, new_szl, sz[i+1:end]..., 1)
		szl = new_szl
	end
	Tensors[end] = permutedims(R, [1, 3, 2])

	for i=1:rank
		tensor = Tensors[i]
		println("$(i)th tensor : ")
		show(stdout, "text/plain", tensor)
		println("\n\n")
	end
end



# LurTensors version
tagvec = ["site$(i)" for i=1:5]
# left dummy leg / right dummy leg
T_withdummy = reshape(T, (1, size(T)..., 1))
Ti = LurTensor(T_withdummy, "ldm", tagvec..., "rdm")

let
	Ri = Ti
	old_bond_tag = "ldm"
	Tensors = Vector{LurTensor}(undef, rank)
	for i=1:rank-1
		# QR decomposition, regard Ri as matrix of dimension
		# (old_bond_idx, idxs[i]) * (other indices of Ri)
		Qi, Ri = qr(Ri, old_bond_tag, tagvec[i]; addtag="$(i)~$(i+1)")
		# Leg between Qi and Ri created in QR decomposition.
		# Used the function which gives common indices of two tensors.
		new_bond_tag = commonind(Qi, Ri).tag
		# Change the order of index described below
		Tensors[i] = permutedims(Qi, [1, 3, 2])
		old_bond_tag = new_bond_tag
	end
	Tensors[end] = permutedims(Ri, [1, 3, 2])

	# The same result as the matrix manipulation version
	for i=1:rank
		tensor = Tensors[i]
		println("$(i)th tensor : ")
		show(stdout, tensor; showarr=true)
		println("\n\n")
	end
end

