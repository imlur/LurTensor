include("../../Tensor/LurTensor.jl")

let
	Nkeep, Nstep = 10, 200
	tau_ini, tau_fin = 1, 0.01
	taus = tau_ini * ((tau_fin / tau_ini) .^ collect(0:Nstep) / Nstep)

	# sl : site,left / sr : site,right
	Sarr = getlocalspace("Spin", 1, 'S')
	S1 = LurTensor(Sarr, ("sl", 1), ("sl", 0), "S")
	S2 = conj(LurTensor(Sarr, ("sr", 1), ("sr", 0), "S"))
	HSS = S1 * S2
	HSS2 = mapprime(HSS * HSS', 2, 1)
	
	H = HSS + HSS2 / 3

	initl1, initl2 = rand(Nkeep, 1), rand(Nkeep, 1)	
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

	(_, _, _, _), Eiter = iTEBD_GS_Vidal!([l1, l2, g1, g2], H, "sl", "sr", taus; Nkeep=Nkeep)
	nothing
end
