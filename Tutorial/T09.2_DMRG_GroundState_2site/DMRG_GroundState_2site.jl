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
	Hs[1] = Hs[1][:, :, end, :]
	Hs[end] = Hs[end][:, :, :, 1]

	# initial MPS for 1site DMRG
	Minit1 = Vector{LurTensor}(undef, L)
	Minit1[1] = LurTensor(rand(1, Nkeep, 2), btags[1], btags[2], stags[1])
	Minit1[end] = LurTensor(rand(Nkeep, 1, 2), btags[end-1], btags[end], stags[end])
	for itN=2:L-1
		Minit1[itN] = LurTensor(rand(Nkeep, Nkeep, 2), btags[itN], btags[itN+1], stags[itN])
	end
	# initial MPS for 2site DMRG. Same to Minit1 
	Minit2 = [copy(t) for t in Minit1]

	E0_exact = 0.5 - 1 / (2 * sin(pi / (2*(L+1))))
	println("----------1site----------")
	E0_1site, _, _ = DMRG_GS_1site!(Minit1, Hs; Nkeep=Nkeep, Nsweep=Nsweep)
	println("Exact : $(E0_exact)")
	println("Diff : $(E0_1site - E0_exact)")
	println("----------2site----------")
	E0_2site, _, _ = DMRG_GS_2site!(Minit2, Hs; Nkeep=Nkeep, Nsweep=Nsweep)
	println("Exact : $(E0_exact)")
	println("Diff : $(E0_2site - E0_exact)")
end
