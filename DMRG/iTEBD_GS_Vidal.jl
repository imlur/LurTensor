iTEBD_GS_Vidal!(l1, l2, g1, g2, H, Hli, Hri, taus; kw...) =
	iTEBD_GS_Vidal!(l1, l2, g1, g2, H, Index(Hli), Index(Hri), taus; kw...)

# Indices of H should be (Hli, Hli', Hri, Hri') (order does not matter)
function iTEBD_GS_Vidal!(l1, l2, g1, g2, H, 
						 Hli::Index, Hri::Index, taus; Nkeep=30)
	Nstep = length(taus)
	Eiter = zeros(Nstep, 2, 2)

	l1g1 = check_common_inds(l1, g1, "l1", "g1")
	l2g1 = check_common_inds(l2, g1, "l2", "g1")
	l1g2 = check_common_inds(l1, g2, "l1", "g2")
	l2g2 = check_common_inds(l2, g2, "l2", "g2")
	bonds = [l1g1, l2g1, l1g2, l2g2]

	# odd site index, even site index
	osi = uniqueind(g1, bonds)
	esi = uniqueind(g2, bonds)
	
	# Reshape to exponentiate
	Hinds = [Hli', Hri', Hli, Hri]
	H = permutedims(H, Hinds)
	Hsz = size(H); rdim = Hsz[1] * Hsz[2]
	Hmat = reshape(H.arr, rdim, rdim)
	evals, evecs = eigen((Hmat + Hmat') / 2)
	# odd bonds 
	expH_oind = [osi', esi', osi, esi]
	expH_eind = [esi', osi', esi, osi]

	for itN=1:Nstep
		tau = taus[itN]
		expH_arr = evecs * Diagonal(exp.(-tau * evals)) * evecs'
		expH_arr = reshape(expH_arr, Hsz...)

		expH_forodd = LurTensor(expH_arr, expH_oind)
		expH_foreven = LurTensor(expH_arr, expH_eind)

		g1, l1, g2 = updatebond(l2, g1, l1, g2, expH_forodd)

		g2, l2, g1 = updatebond(l1, g2, l2, g1, expH_foreven)
	end
	
	return (l1, l2, g1, g2), Eiter
end

# ls : lambda, at side/ lc : lambda, at center (at figure in L10.2, page 3)
# gl : gamma, at left / gr : gamma, at right
function updatebond(ls, gl, lc, gr, expH)
	
	return gl, lc, gr
end

function calcenergy()

end
