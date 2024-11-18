include("../../Tensor/LurTensor.jl")

let
	J1, J2 = 1, 0.5; L = 40
	# Bond dimension, number of pairs of left + right sweeps
	Nkeep = 30; Nsweep = 5; 
	tol = Nkeep * 100 * eps(1.) # Numerical tolerance for degeneracy

	# define tags
	stags = ["site,$(i)" for i=1:L] # Site tag
	hbtags = ["hld", ["h$(i)~$(i+1)" for i=1:L-1]..., "hrd"] # Bond leg for hamiltonian
	btags = ["ld", ["$(i)~$(i+1)" for i=1:L-1]..., "rd"] # Bond leg tag for state
	Sarr = getlocalspace("Spin", 0.5, 'S')

	# MPO representation of Hamiltonian
	Hloc = zeros(2, 2, 8, 8)
	Hloc[:, :, 1, 1] = Hloc[:, :, 8, 8] = [1. 0.; 0. 1.]
	Hloc[:, :, 5, 2] = Hloc[:, :, 6, 3] = Hloc[:, :, 7, 4] = [1. 0.; 0. 1.]
	Hloc[:, :, 2:4, 1] = Sarr
	Hloc[:, :, 8, 2:4] = permutedims(Sarr, [2, 1, 3]) * J1
	Hloc[:, :, 8, 5:7] = permutedims(Sarr, [2, 1, 3]) * J2

	Hs = Vector{LurTensor}(undef, L)
	for i=1:L
		Hs[i] = LurTensor(Hloc, (stags[i], 1), (stags[i], 0), hbtags[i], hbtags[i+1])
	end

	# Initialization
	Hs[1] = Hs[1][:, :, end, :]
	Hs[end] = Hs[end][:, :, :, 1]

	# initial MPS for 1site DMRG
	Minit = Vector{LurTensor}(undef, L)
	Minit[1] = LurTensor(rand(1, Nkeep, 2), btags[1], btags[2], stags[1])
	Minit[end] = LurTensor(rand(Nkeep, 1, 2), btags[end-1], btags[end], stags[end])
	for itN=2:L-1
		Minit[itN] = LurTensor(rand(Nkeep, Nkeep, 2), btags[itN], btags[itN+1], stags[itN])
	end

	E0_exact = -3 * L / 8
	E0, _, _ = DMRG_GS_2site!(Minit, Hs; Nkeep=Nkeep, Nsweep=Nsweep)
	println("Exact : $(E0_exact)")
	println("Diff : $(E0 - E0_exact)")
	for i=1:8
		display(Minit[i])
		# Alternating bond dimensions (1, 2, 1, 2, ...)
	end
end
