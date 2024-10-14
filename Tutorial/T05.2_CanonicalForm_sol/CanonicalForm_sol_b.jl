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
	
	M, _, _ = canonform(M, N, "ld", "rd")
	M, _, _ = canonform(M, 0, "ld", "rd")

	# right to the left
	nkeeps = 5:D; Ss = zeros(Float64, length(nkeeps))
	for i = 1:length(nkeeps)
		_, s, _ = canonform(copy(M), N, "ld", "rd"; nkeep=nkeeps[i])
		# s is 1 * 1 LurTensor, so its first element means its value
		Ss[i] = s[1]
	end
	println(Ss)

	# right to the right
	for i = 1:length(nkeeps)
		_, s, _ = canonform(copy(M), 0, "ld", "rd"; nkeep=nkeeps[i])
		Ss[i] = s[1]
	end
	println(Ss)

	# right to the bond
	for i = 1:length(nkeeps)
		_, s, _ = canonform(copy(M), 25, "ld", "rd"; nkeep=nkeeps[i])
		Ss[i] = norm(s)
	end
	println(Ss)
end
