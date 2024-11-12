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

	for h in Hs
		display(h)
	end
end
