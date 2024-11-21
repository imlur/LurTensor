include("../../Tensor/LurTensor.jl")
using Plots

let
	Nkeep, Nstep = 30, 2000
	tau_ini, tau_fin = 1, 1e-6
	taus = tau_ini * ((tau_fin / tau_ini) .^ (collect(0:Nstep) / Nstep))

	# sl : site,left / sr : site,right
	Sarr = getlocalspace("Spin", 1, 'S')
	S1 = LurTensor(Sarr, ("sl", 1), ("sl", 0), "S")
	S2 = conj(LurTensor(Sarr, ("sr", 0), ("sr", 1), "S"))
	H = S1 * S2

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
	
	Eexact = -1.401484039

	p1 = plot()
    p1 = plot!(Eodd[1:60] .- Eexact)
    p1 = plot!(Eeven[1:60] .- Eexact)
	display(p1)

	Eaver = (Eodd + Eeven) / 2
	p2 = plot()
	p2 = plot!(Eaver .- Eexact, yaxis=:log)
	display(p2)

	# Transfer matrix
	T = addtags(g1, "left"; tag="l2") * l1
	T *= g2; T *= addtags(l2, "right"; tag="g1")
	removetags!(T, "l2,g1")

	W = T * prime(T', -1; tag="Site")
	W = permutedims(W, "left", ("left", 1), "right", ("right", 1))
	sz = prod(size(W)[1:2])

	MW = reshape(W.arr, sz, sz)
	vals, vecs = eigen(MW)
	display(MW)
	println("Eigenvalues of transfer operator : ")
	display(vals)
	println("Eigenvector corresponds to largest eigenvalue (=1) :")
	display(reshape(vecs[:, end], size(W)[1:2]...))
	
	# TODO: Do exercise b (change tau_fin, Nstep, Nkeep and see convergence)
end
