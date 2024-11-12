include("../../Tensor/LurTensor.jl")

sz = [2 3 2 3 4]
T = reshape((1:prod(sz)), sz...)
T = T / norm(T)
# rank of the tensor
rank = length(sz)

function ee_from_svals(svals)
	return -sum([s==0 ? 0 : s^2*log2(s^2) for s in svals])
end

let
	# Decomposition results
	Tensors = Vector{Array{Float64}}(undef, rank)
	Sent = zeros(rank-1)
	R = T; szl = 1
	for i=1:rank-1
		R = reshape(R, szl*sz[i], prod(sz[i+1:end]))
		# svd results. U, Vt : matrix, S : vector of singular values
		U, S, V = svd(R)

		# Truncate singular values smaller than eps
		# Number of nonzero singular values
		nnzs = sum(S .> eps(1.0))
		U, S, V = U[:, 1:nnzs], S[1:nnzs], V[:, 1:nnzs]
		new_szl = nnzs
		U = reshape(U, szl, sz[i], new_szl)

		# Save results, and ready for next iteration
		Tensors[i] = permutedims(U, [1 3 2])
		Sent[i] = -sum((S.^2).*log2.(S.^2))
		R = reshape(diagm(S) * V', new_szl, sz[i+1:end]..., 1)
		szl = new_szl
	end
	Tensors[end] = permutedims(R, [1 3 2])

	# Print resultant tensors
	for i=1:rank
		tensor = Tensors[i]
		println("$(i)th tensor : ")
		show(stdout, "text/plain", tensor)
		println("\n")
	end

	# Print entanglement entropy
	println("Entanglement entropy : ")
	for i=1:rank-1
		print("$(round(Sent[i], digits=5))  ")
	end
	println("\n\n")
end

# LurTensors version
tagvec = ["site$(i)" for i=1:5]
# left dummy leg / right dummy leg
T_withdummy = reshape(T, (1, size(T)..., 1))
# ldm : left dummy leg, rdm : right dummy leg
Ti = LurTensor(T_withdummy, "ldm", tagvec..., "rdm")

let
	Ri = Ti
	old_bond_tag = "ldm"
	Sent = zeros(rank - 1)
	Tensors = Vector{LurTensor}(undef, rank)
	for i=1:rank-1
		# 'cutoff' option truncate small singular values automatically 
		atag = "$(i)~$(i+1)"
		(Ui, Si, Vi), _ = svd(Ri, old_bond_tag, tagvec[i]; addtag=atag, cutoff=eps(1.0))
		new_bond_tag = ind(Si; tag="left,$(atag)").tag
		Tensors[i] = removetags(permutedims(Ui, [1, 3, 2]), "svd,left")
		svals = diag(Si); Sent[i] = -sum((svals.^2).*log2.(svals.^2))

		# Prepare for next iteration
		old_bond_tag = new_bond_tag
		Ri = Si * Vi
	end
	Tensors[end] = removetags(permutedims(Ri, [1, 3, 2]), "svd,left")

	for i=1:rank
		tensor = Tensors[i]
		println("$(i)th tensor : ")
		println(tensor)
		println()
	end

	for i=1:rank-1
		print("$(round(Sent[i], digits=5))  ")
	end
	println()

	# Contract tensors to get the original tensor
	T2 = Tensors[1]
	for i=2:rank
		T2 = T2 * Tensors[i]
	end
	# Norm difference. about 1e-15
	println("Norm of T - T2 is $(norm(Ti - T2))")
end
