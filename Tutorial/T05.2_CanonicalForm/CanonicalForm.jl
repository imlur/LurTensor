include("../../Tensor/LurTensor.jl")

let
	N, d, D = 50, 3, 30

	M = Vector{LurTensor}(undef, N)
	site_tags = ["site,$(i)" for i=1:N]
	bond_tags = ["ld", ["$(i)~$(i+1)" for i=1:N-1]..., "rd"]

	for i=1:N
		if i == 1
			M[i] = LurTensor(rand(1, D, d), "ld", bond_tags[2], site_tags[1])
		elseif i == N
			M[i] = LurTensor(rand(D, 1, d), bond_tags[end-1], "rd", site_tags[end])
		else
			M[i] = LurTensor(rand(D, D, d), bond_tags[i], bond_tags[i+1], site_tags[i])
		end
	end

	# Without copy, original M vector is altered
	M_L, S_L, _ = canonform(copy(M), N, "ld", "rd")
	show(stdout, S_L; showarr=true)

	# Overlap between the transformed MPS (S pulled out) and the original one
	Tovl = LurTensor(reshape([1], 1, 1), ["ld", "ld"], [0, 1])
	for i=1:N
		Tovl = updateleft(Tovl, M_L[i]', (site_tags[i], 1), M[i], (site_tags[i], 0))
	end
	show(stdout, Tovl / S_L)
	println()

	# Use updateleft to "close the zipper" from right
	Tovl = LurTensor(reshape([1], 1, 1), ["rd", "rd"], [0, 1])
	for i=N:-1:1
		Tovl = updateleft(Tovl, M_L[i]', (site_tags[i], 1), M[i], (site_tags[i], 0))
	end
	show(stdout, Tovl / S_L)
	println()

	# Overlap (in right canonical form)
	M_R, S_R, _ = canonform(copy(M), 0, "ld", "rd")

	Tovl = LurTensor(reshape([1], 1, 1), ["ld", "ld"], [0, 1])
	for i=1:N
		Tovl = updateleft(Tovl, M_R[i]', (site_tags[i], 1), M[i], (site_tags[i], 0))
	end
	show(stdout, Tovl / S_R)
	println()

	# Overlap (in bond-canonical form)
	M_B, S_B, _ = canonform(copy(M), 25, "ld", "rd")
	Tovl = LurTensor(reshape([1], 1, 1), ["ld", "ld"], [0, 1])
	for i=1:N
		Tovl = updateleft(Tovl, M_B[i]', (site_tags[i], 1), M[i], (site_tags[i], 0))
		if i == 25
			Tovl = Tovl * S_B'
		end
	end
	show(stdout, Tovl / S_L^2)
	println()
end
