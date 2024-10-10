include("../../Tensor/LurTensor.jl")

sz = [2 3 2 3 4]
T = reshape((1:prod(sz)), sz...)
T = T / norm(T)
# rank of the tensor
rank = length(sz)

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

	# Contract tensors to get the original tensor
	T2 = Tensors[1]
	for i=2:rank
		T2 = T2 * Tensors[i]
	end
	show(stdout, Ti)
	show(stdout, T2)
	# about e-15
	println("Norm of T - T2 is $(norm(Ti - T2))")
end


