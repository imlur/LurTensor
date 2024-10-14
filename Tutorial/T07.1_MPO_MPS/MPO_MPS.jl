include("../../Tensor/LurTensor.jl")

J = 1; L = 6; M = Vector{LurTensor}(undef, L)
stags = ["site,$(i)" for i=1:L] # Site tag
hbtags = ["hld", ["h$(i)~$(i+1)" for i=1:L]..., "hrd"] # Bond leg for hamiltonian
btags = ["ld", ["$(i)~$(i+1)" for i=1:L-1]..., "rd"] # Bond leg tag for state

let
	E_G = 0
	Hprev = LurTensor(reshape([0], 1, 1), ["ld", "ld"], [0, 1])
	Aprev = LurTensor(reshape([1], 1, 1), ["ld", "ld"], [0, 1])
	Sprev = LurTensor()
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
			E_G = diag(D)[1]
		else
			M[i] = Anow
		end

		Aprev = Anow
		Hprev = Hnow
		left_dim *= 2
	end
	println(E_G)
end

