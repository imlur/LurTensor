include("../../Tensor/LurTensor.jl")

let
	J = -1; L = 40
	# Bond dimension, number of pairs of left + right sweeps
	Nkeep = 30; Nsweep = 4; 
	tol = Nkeep * 100 * eps(1.) # Numerical tolerance for degeneracy

	# define tags
	stags = ["site,$(i)" for i=1:L] # Site tag
	hbtags = ["hld", ["h$(i)~$(i+1)" for i=1:L-1]..., "hrd"] # Bond leg for hamiltonian
	btags = ["ld", ["$(i)~$(i+1)" for i=1:L-1]..., "rd"] # Bond leg tag for state
	Sarr = getlocalspace("Spin", 0.5, 'S')

	# MPO representation of Hamiltonian
	Hloc = zeros(2, 2, 4, 4)
	Hloc[:, :, 1, 1] = Hloc[:, :, 4, 4] = [1. 0.; 0. 1.]
	Hloc[:, :, 2, 1] = Sarr[:, :, 1]
	Hloc[:, :, 3, 1] = Sarr[:, :, 3]
	Hloc[:, :, 4, 2] = Sarr[:, :, 3] * J
	Hloc[:, :, 4, 3] = Sarr[:, :, 1] * J

	Hs = Vector{LurTensor}(undef, L)
	for i=1:L
		Hs[i] = LurTensor(Hloc, (stags[i], 1), (stags[i], 0), hbtags[i], hbtags[i+1])
	end
	
	# Initialization
	Hs[1] = Hs[1][:, :, end:end, :]
	Hs[end] = Hs[end][:, :, :, 1:1]

	Minit = Vector{LurTensor}(undef, L)
	Hprev = LurTensor(reshape([1], 1, 1), ["ld", "ld"], [0, 1])
	Aprev = LurTensor(reshape([1], 1, 1), ["ld", "ld"], [0, 1])
	left_dim = 1; left_tag = "ld"

	# Initializae MPS ansatz with iterative diagonalization
	for i=1:L
		Anow = getidentity([left_dim, 2], left_tag, stags[i], "$(btags[i+1]),temp")
		Hnow = updateleft(Hprev, Anow', Hs[i], Anow)
			
		Hmat = subtensor(Hnow, [hbtags[i+1], "hld"], [1, 1])
		# Diagonalize and truncate (considering degeneracy)
		(V, D, Vd), _ = eigen((Hmat + hconj(Hmat)) / 2, ("$(btags[i+1]),temp", 1))
		Dd = diag(D)
		if i == L
			# Last iteration -> pick only ground state
			Ntr = 1
		elseif length(Dd) > Nkeep
			Ntr = findfirst(x -> x > Dd[Nkeep + 1] - tol, Dd) - 1
		else
			Ntr = length(Dd)
		end

		Vtrunc = LurTensor(V[:, 1:Ntr], "$(btags[i+1]),temp", btags[i+1]) 
		Anow = Anow * Vtrunc
		Minit[i] = Anow

		Hprev = (Hnow * Vtrunc) * Vtrunc'
		Aprev = Anow; left_dim = Ntr; left_tag = btags[i+1]
	end
	# Print MPS
	for m in Minit
		display(m)
	end
	# Print Hamiltonian
	for h in Hs
		display(h)
	end
end
