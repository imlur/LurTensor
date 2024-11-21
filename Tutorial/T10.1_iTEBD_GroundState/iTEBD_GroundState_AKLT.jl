include("../../Tensor/LurTensor.jl")
using Plots

let
	Nkeep, Nstep = 10, 200
	tau_ini, tau_fin = 1, 0.01
	taus = tau_ini * ((tau_fin / tau_ini) .^ (collect(0:Nstep) / Nstep))

	# sl : site,left / sr : site,right
	Sarr = getlocalspace("Spin", 1, 'S')
	S1 = LurTensor(Sarr, ("sl", 1), ("sl", 0), "S")
	S2 = conj(LurTensor(Sarr, ("sr", 0), ("sr", 1), "S"))
	HSS = S1 * S2
	HSS2 = mapprime(HSS * HSS', 2, 1)
	
	H = HSS + HSS2 / 3

	initl1, initl2 = rand(Nkeep), rand(Nkeep)	
	l1 = LurTensor(Diagonal(initl1), "l1,g1", "l1,g2")
	l2 = LurTensor(Diagonal(initl2), "l2,g2", "l2,g1")
	garr1, garr2 = rand(Nkeep, Nkeep, 3), rand(Nkeep, Nkeep, 3)
	g1 = LurTensor(garr1, "l2,g1", "l1,g1", "Site,odd")
	g2 = LurTensor(garr2, "l1,g2", "l2,g2", "Site,even")

	# Tensor order is same to original MATLAB code
	# -> g1 -> l1 -> g2 -> l2 ->
	#    |			 |
	# Site,odd    Site,even

	# bond tag names are from names of tensors which are connected to the leg

	(l1, l2, g1, g2), Eiter = iTEBD_GS_Vidal(l1, l2, g1, g2, H, "sl", "sr", taus; Nkeep=Nkeep)
	Eiter = permutedims(Eiter, [2, 1, 3])
	Eodd = Eiter[:, :, 1][:]
	Eeven = Eiter[:, :, 2][:]
	
	Eexact = -2 / 3
	p = plot()
    p = plot!(Eodd[1:20] .- Eexact, yaxis=:log)
    p = plot!(Eeven[1:20] .- Eexact, yaxis=:log)
	display(p)

	A1 = l2 * g1
	A1l = A1 * prime(A1, 1; tag="g2")
	A1r = A1 * prime(A1, 1; tag="l1")
	println("Check left/right normalization of l2 * g1")
	display(A1l.arr - Matrix(I, 2, 2))
	display(A1r.arr - Matrix(I, 2, 2))

	println("Check singular values are 1/sqrt(2)")
	display(l1.arr - Diagonal([1/sqrt(2), 1/sqrt(2)]))	
	display(l2.arr - Diagonal([1/sqrt(2), 1/sqrt(2)]))	

	T = addtags(g1, "left"; tag="l2") * l1
	T *= g2; T *= addtags(l2, "right"; tag="g1")
	removetags!(T, "l2,g1")

	W = T * prime(T', -1; tag="Site")
	W = permutedims(W, "left", ("left", 1), "right", ("right", 1))
	sz = prod(size(W)[1:2])

	MW = reshape(W.arr, sz, sz)
	vals, vecs = eigen(MW)
	println("Eigenvalues of transfer operator : ")
	display(sqrt.(vals))
	println("Eigenvector corresponds to largest eigenvalue (=1) :")
	display(reshape(vecs[:, end], 2, 2))
end
