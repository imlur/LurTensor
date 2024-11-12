include("../../Tensor/LurTensor.jl")

# Solution to Exercise a
let
	L = 4

	stags = ["site,$(i)" for i=1:L] # Site tag
	hbtags = ["hld", ["h$(i)~$(i+1)" for i=1:L-1]..., "hrd"] # Bond leg for hamiltonian
	btags = ["ld", ["$(i)~$(i+1)" for i=1:L-1]..., "rd"] # Bond leg tag for state
	mtags = ["mld", ["m$(i)~$(i+1)" for i=1:L-1]..., "mrd"] # tags for HM
	Sarr = getlocalspace("Spin", 1, 'S')

	Hloc = zeros(3, 3, 14, 14)
	Hloc[:, :, 1, 1] = Hloc[:, :, 14, 14] = Matrix(I, 3, 3)
	Hloc[:, :, 2:4, 1] = Sarr
	Hloc[:, :, 14, 2:4] = permutedims(Sarr, [2, 1, 3])

	i = 5
	for a=1:3
		for b=1:3
			Hloc[:, :, i, 1] = Sarr[:, :, 4-a] * Sarr[:, :, 4-b] 
			Hloc[:, :, 14, i] = Sarr[:, :, a] * Sarr[:, :, b] / 3
			i += 1
		end
	end

	# By MPO
	Hs = Vector{LurTensor}(undef, L)
	for i=1:L
		Hs[i] = LurTensor(Hloc, (stags[i], 1), (stags[i], 0), hbtags[i], hbtags[i+1])
	end
	Hs[1] = Hs[1][:, :, end:end, :]
	Hs[end] = Hs[end][:, :, :, 1:1]

	Hs_tot = Hs[1]
	for i=2:L
		Hs_tot = Hs_tot * Hs[i]
	end
	site_inds = [Index("site,$(i)", 0) for i=1:L]
	Hs_tot = mergelegs(Hs_tot, "hld", (site_inds')..., "hrd", ("rd", 1))
	Hs_tot = mergelegs(Hs_tot, site_inds..., ("rd", 0))

	# Constructed iteratively
	S1 = LurTensor(Sarr, ("Site", 1), "Temp", "S1")
	S2 = LurTensor(Sarr, "Temp", "Site", "S2")
	SS = permutedims(S1 * S2, ("Site", 1), "Site", "S1", "S2")

	S1dag = LurTensor(Sarr, "Temp", ("Site", 1), "S1")
	S2dag = LurTensor(Sarr, "Site", "Temp", "S2")
	SSdag = permutedims(conj(S1dag) * conj(S2dag), ("Site", 1), "Site", "S1", "S2")

	Hprev = LurTensor(reshape([0], 1, 1), ["ld", "ld"], [0, 1])
	Aprev = LurTensor(reshape([1], 1, 1), ["ld", "ld"], [0, 1])
	Anow = LurTensor(); Hnow = LurTensor(); 
	Sprev = LurTensor(); S2prev = LurTensor()
	left_dim = 1

	for i=1:L
		# Rank-3 identity tensor for the current iteration
		Anow = getidentity([left_dim, 3], btags[i], stags[i], btags[i+1])
		# contract the Hamiltonian up to the last iteration with
		# ket and bra tensors
		Hnow = updateleft(Hprev, Anow', (stags[i], 1), Anow, (stags[i], 0))

		if i > 1
			# Heisenberg interaction
			Sdag = LurTensor(Sarr, (stags[i], 0), (stags[i], 1), "S")
			SSdag_ = LurTensor(SSdag, (stags[i], 1), (stags[i], 0), "S1", "S2")
			Hsp = updateleft(Sprev, Anow', Sdag, Anow)
			Hsp2 = updateleft(S2prev, Anow', SSdag_, Anow)
			Hnow = Hnow + Hsp + Hsp2 / 3
		end

		S = LurTensor(Sarr, (stags[i], 1), (stags[i], 0), "S")
		SS_ = LurTensor(SS, (stags[i], 1), (stags[i], 0), "S1", "S2")
		Sprev = updateleft(Anow', (btags[i], 1), S, Anow, (btags[i], 0))
		S2prev = updateleft(Anow', (btags[i], 1), SS_, Anow, (btags[i], 0))

		Aprev = Anow; Hprev = Hnow; left_dim *= 3
	end
	println(norm(Hs_tot - Hnow))



	# Solution to Exercise b, define tags again since system size is changed
	L = 50
	stags = ["site,$(i)" for i=1:L] # Site tag
	hbtags = ["hld", ["h$(i)~$(i+1)" for i=1:L-1]..., "hrd"] # Bond leg for hamiltonian
	btags = ["ld", ["$(i)~$(i+1)" for i=1:L-1]..., "rd"] # Bond leg tag for state
	mtags = ["mld", ["m$(i)~$(i+1)" for i=1:L-1]..., "mrd"] # tags for middle legs

	# Generate AKLT state 
	AKLT = zeros(2, 2, 3)

	# Local spin S_z = +1
	AKLT[1, 2, 1] = sqrt(2/3)
	# Local spin S_z = 0
	AKLT[1, 1, 2] = -1 / sqrt(3)
	AKLT[2, 2, 2] = 1 / sqrt(3)
	# Local spin S_z = -1
	AKLT[2, 1, 3] = -sqrt(2/3)

	# Get left-canonical form of AKLT state with specific boundary condition
	function get_normalized_AKLT(L, alpha, beta)
		M = Vector{LurTensor}(undef, L)
		for i=1:L
			T = LurTensor(AKLT, btags[i], btags[i+1], stags[i])
			if i == 1
				M[i] = T[alpha:alpha, :, :]
			elseif i == L
				M[i] = T[:, beta:beta, :]
			else
				M[i] = T
			end
		end
		M, _, _ = canonform(M, L, "ld", "rd")
		return M
	end

	Hs = Vector{LurTensor}(undef, L)
	for i=1:L
		Hs[i] = LurTensor(Hloc, (stags[i], 1), (stags[i], 0), hbtags[i], hbtags[i+1])
	end
	Hs[1] = Hs[1][:, :, end:end, :]
	Hs[end] = Hs[end][:, :, :, 1:1]
	for a=1:2
		for b=1:2
			M = get_normalized_AKLT(L, a, b)

			# get H|AKLT>
			HM = Vector{LurTensor}(undef, L)
			for i=1:L
				MHs = mergelegs(M[i] * Hs[i], btags[i], hbtags[i], mtags[i])
				HM[i] = mergelegs(MHs, btags[i+1], hbtags[i+1], mtags[i+1])
			end
			
			HM, HMnorm, _ = canonform(HM, 0, "mld", "mrd")
			HM, _, _ = canonform(HM, L, "mld", "mrd")

			HM[1] = HM[1] * HMnorm
			MHM = LurTensor(reshape([1], 1, 1), ["mld", "ld"], [0, 1])
			for i=1:L
				MHM = updateleft(MHM, M[i]', (stags[i], 1), HM[i], (stags[i], 0))
			end
			println("for alpha = $(a), beta = $(b),")
			println("<H> = $(value(MHM)^2)")
			println("<H>^2 - <H^2> = $(value(MHM)^2 - value(HMnorm)^2)")
		end
	end
end
