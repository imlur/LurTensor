include("../../Tensor/LurTensor.jl")

N = 50

# Uniform hopping, logarithmic hopping
ts = [[1 for i=1:N-1], 2. .^(-(0:N-2)/2)]

Nkeep = 300; 
tol = Nkeep * 100 * eps(1.) # Numerical tolerance for degeneracy
stag = ["site,$(i)" for i=1:N] # site tag

function iter_diag(ts::Vector)
	# Tensors for the vaccum (i.e., dummy leg)
	Hprev = LurTensor(reshape([0], 1, 1), ["ldm", "ldm"], [0, 1])
	Aprev = LurTensor(reshape([1], 1, 1), ["ldm", "ldm"], [0, 1])
	Fprev = LurTensor()
	left_dim = 1; left_tag = "ldm"

	# Ground-state energies for different lengths
	E0_iter = zeros(1, N); E0_exact = zeros(1, N)

	for i=1:N
		Anow = getidentity([left_dim, 2], left_tag, stag[i], "$(i),temp")
		Hnow = updateleft(Hprev, Anow', (stag[i], 1), Anow, (stag[i], 0))
		
		if i > 1
			Z = getlocalspace("Fermion", 'Z', (stag[i], 1), (stag[i], 2))
			F = getlocalspace("Fermion", 'F', (stag[i], 2), (stag[i], 0))
			ZF = Z * F
			Hhop = -ts[i-1] * updateleft(Fprev, Anow', hconj(ZF), Anow)
			Hnow = Hnow + Hhop + hconj(Hhop)
		end
		
		(V, D, Vd), _ = eigen((Hnow + hconj(Hnow)) / 2, ("$(i),temp", 1))
		Dd = diag(D) # diagonal elements of D (sorted eigenvalues)
		E0_iter[i] = Dd[1]
		Etr = Dd[min(length(Dd), Nkeep)]
		oks = Dd .< (Etr + tol)

		FF = getlocalspace("Fermion", 'F', (stag[i], 1), (stag[i], 0))
		Vtrunc = LurTensor(V[:, oks], "$(i),temp", "$(i)") 
		Aprev = Anow * Vtrunc
		Hprev = LurTensor(D[oks, oks], ("$(i)", 1), ("$(i)", 0))
		Fprev = updateleft(Aprev', (left_tag, 1), FF, Aprev, (left_tag, 0))

		left_dim = size(Hprev)[1]; left_tag = "$(i)"

		if i > 1
			E0_exact[i] = nonIntTB(ts[1:i-1])[1]
		end
	end
	return E0_iter, E0_exact
end

# TODO: Plot them

# uniform hopping
i, e = iter_diag(ts[1])
println(i - e) # ground-state energy error

# logarithmic hopping
i, e = iter_diag(ts[2])
println(i - e) # ground-state energy error
