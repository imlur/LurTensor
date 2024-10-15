include("../../Tensor/LurTensor.jl")

J = 1; L = 6; 
stags = ["site,$(i)" for i=1:L] # Site tag
hbtags = ["hld", ["h$(i)~$(i+1)" for i=1:L-1]..., "hrd"] # Bond leg for hamiltonian
btags = ["ld", ["$(i)~$(i+1)" for i=1:L-1]..., "rd"] # Bond leg tag for state

let
	M = Vector{LurTensor}(undef, L)
	E_G = 0
	Hprev = LurTensor(reshape([0], 1, 1), ["ld", "ld"], [0, 1])
	Aprev = LurTensor(reshape([1], 1, 1), ["ld", "ld"], [0, 1])
	Sprev = LurTensor(); Hnow = LurTensor()
	left_dim = 1

	for i=1:L
		Anow = getidentity([left_dim, 2], btags[i], stags[i], btags[i+1])
		Hnow = updateleft(Hprev, Anow', (stags[i], 1), Anow, (stags[i], 0))

		if i > 1
			S_dag = getlocalspace("Spin", 0.5, 'S', (stags[i], 0), (stags[i], 1), "S")
			Hsp = updateleft(Sprev, Anow', conj(S_dag), Anow)
			Hnow = Hnow + J * Hsp
		end

		S = getlocalspace("Spin", 0.5, 'S', (stags[i], 1), (stags[i], 0), "S")
		Sprev = updateleft(Anow', (btags[i], 1), S, Anow, (btags[i], 0))

		if i == L
			# eigenvalues (diagonal of D) are already sorted
			(V, D, Vd), _ = eigen((Hnow + hconj(Hnow)) / 2, (btags[end], 1); hermitian=true)
			vgs = LurTensor(V[:, 1:1], ("rd", 0), ("rd", 1))
			M[i] = noprime(Anow * vgs)
			E_G = diag(D)[1]
		else
			M[i] = Anow
		end

		Aprev = Anow
		Hprev = Hnow
		left_dim *= 2
	end
	for m in M
		println(m)
	end
	println("-------------------------")

	M, _, _ = canonform(M, 0, "ld", "rd")
	for m in M
		println(m)
	end
	println(E_G)

	# MPO representation of Heisenberg chain Hamiltonian
	Hloc = zeros(2, 2, 5, 5)
	Sarr = getlocalspace("Spin", 0.5, 'S')
	Hloc[:, :, 1, 1] = Hloc[:, :, 5, 5] = [1. 0.; 0. 1.]
	Hloc[:, :, 2:4, 1] = Sarr
	Hloc[:, :, 5, 2:4] = permutedims(Sarr, [2, 1, 3]) * J

	Hs = Vector{LurTensor}(undef, L)
	for i=1:L
		Hs[i] = LurTensor(Hloc, (stags[i], 1), (stags[i], 0), hbtags[i], hbtags[i+1])
	end
	Hs[1] = Hs[1][:, :, end:end, :]
	Hs[end] = Hs[end][:, :, :, 1:1]

	# Check whether the MPO construction is right
	Hs_tot = Hs[1]
	for i=2:L
		Hs_tot = Hs_tot * Hs[i]
	end
	println(Hnow)
	site_inds = [Index("site,$(i)", 0) for i=1:L]
	Hs_tot = mergelegs(Hs_tot, "hld", (site_inds')..., "hrd", ("rd", 1))
	Hs_tot = mergelegs(Hs_tot, site_inds..., ("rd", 0))
	println("norm(Hnow - Hs_tot) = ")
	println(norm(Hnow - Hs_tot)) # Operator norm of different is zero

	println("M : ")
	println(M)
	println("H : ")
	println(Hs)

	# merged bond leg tags
	mtags = ["mld", ["m$(i)~$(i+1)" for i=1:L-1]..., "mrd"] 
	HM = Vector{LurTensor}(undef, L)
	Aleft, Aright = LurTensor(), LurTensor()
	for i=1:L
		MHs = mergelegs(M[i] * Hs[i], btags[i], hbtags[i], mtags[i])
		HM[i] = mergelegs(MHs, btags[i+1], hbtags[i+1], mtags[i+1])
	end
	println("-------------Canonical Form, first time------------")
	HM, HMnorm, _ = canonform(HM, L, "mld", "mrd")
	println(HM)
	show(stdout, HMnorm; showarr=true)

	println("-------------Canonical Form, second time-----------")
	HM, HMnorm2, _ = canonform(HM, 0, "mld", "mrd")
	println(HM)
	show(stdout, HMnorm2; showarr=true)

	println("HMnorm - abs(E_G) = ")
	println(HMnorm[1] - abs(E_G))

	HM[1] = HM[1] * HMnorm
	MHM = LurTensor(reshape([1], 1, 1), ["mld", "ld"], [0, 1])
	for i=1:L
		MHM = updateleft(MHM, M[i]', (stags[i], 1), HM[i], (stags[i], 0))
	end
	println("MHM - E_G = ")
	println(MHM[1] - E_G)
end
