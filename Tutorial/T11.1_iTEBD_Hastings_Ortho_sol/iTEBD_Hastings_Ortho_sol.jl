include("../../Tensor/LurTensor.jl")
using Plots

function check_ortho(gamma, lambda, tempind)
	M = gamma * lambda
	M = M * replaceind(M, tempind, tempind')
	display(M.arr[1:5, 1:5])
	println("norm(M - I) = $(norm(M.arr - Matrix(I, 30, 30)))")
	println()
end

let
	# TODO: after complete ortho function, set Nstep to 2000
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

	# iTEBD result
	(l1, l2, g1, g2), Eiter = iTEBD_GS_Vidal(l1, l2, g1, g2, H, "sl", "sr", taus; Nkeep=Nkeep)
	
	# No need to merge two physical legs
	Gamma2 = (g1 * l1) * g2
	# Left leg of Gamma2
	li = commonind(g1, l2)
	ri = commonind(g2, l2)

	# lo : lambda_ortho, go : gamma_ortho
	lo, go = ortho_orus(l2, Gamma2, li)

	tempind = Index("__temp__", 7373)
	go1 = replaceind(go, li, tempind)
	go2 = replaceind(go, ri, tempind)
	ga1 = replaceind(Gamma2, li, tempind)
	ga2 = replaceind(Gamma2, ri, tempind)

	# Check left-normalized / right-normalized (after orthonormalization)
	check_ortho(go1, lo, tempind)
	check_ortho(go2, lo, tempind)
	# Check left-normalized / right-normalized (before orthonormalization)
	check_ortho(ga1, l2, tempind)
	check_ortho(ga2, l2, tempind)

	tli = Index("trans,left")
	tri = Index("trans,right")
	Gamma_ortho = replaceind(go, li, tli)
	M = Gamma_ortho * lo
	replaceind!(M, li, tri)
	T = M * replaceinds(M, [tli, tri], [tli', tri'])

	T = permutedims(T, [tli, tli', tri, tri'])
	sz = size(lo)[1]
	Tmat = reshape(T.arr, sz^2, sz^2)
	println("30 Largest eigenvalues of transfer operator is ")
	display(real(eigvals(Tmat))[end:-1:end-29])

	# TODO: Complete code to get correlation length
end
