iTEBD_GS_Vidal(l1, l2, g1, g2, H, Hli, Hri, taus; kw...) =
	iTEBD_GS_Vidal(l1, l2, g1, g2, H, Index(Hli), Index(Hri), taus; kw...)

# Indices of H should be (Hli, Hli', Hri, Hri') (order does not matter)
function iTEBD_GS_Vidal(l1, l2, g1, g2, H, 
						 Hli::Index, Hri::Index, taus; Nkeep=10)
	Nstep = length(taus)
	Eiter = zeros(Nstep, 2, 2)
	l1 = LurTensor(Diagonal(l1.arr), l1.inds)
	l2 = LurTensor(Diagonal(l2.arr), l2.inds)
	l1 = l1 / norm(l1)
	l2 = l2 / norm(l2)

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
	Hoind = [osi', esi', osi, esi]
	Heind = [esi', osi', esi, osi]

	for itN=1:Nstep
		tau = taus[itN]
		expH_arr = evecs * Diagonal(exp.(-tau * evals)) * evecs'
		expH_arr = reshape(expH_arr, Hsz...)

		H_forodd = LurTensor(H.arr, Hoind)
		H_foreven = LurTensor(H.arr, Heind)
		expH_forodd = LurTensor(expH_arr, Hoind)
		expH_foreven = LurTensor(expH_arr, Heind)

		# Update odd bonds
		g1, l1, g2 = updatebond(l2, g1, l1, g2, expH_forodd, osi; Nkeep)
		Eodd, Eeven = getenergy(l2, g1, l1, g2, H_forodd, H_foreven, bonds)
		l1 = l1 / norm(l1)
		l2 = l2 / norm(l2)
		Eiter[itN, 1, :] = [Eodd, Eeven]

		# Update even bonds
		g2, l2, g1 = updatebond(l1, g2, l2, g1, expH_foreven, esi; Nkeep)
		Eodd, Eeven = getenergy(l2, g1, l1, g2, H_forodd, H_foreven, bonds)
		l1 = l1 / norm(l1)
		l2 = l2 / norm(l2)
		Eiter[itN, 2, :] = [Eodd, Eeven]

		if itN % 100 == 0
			println("Iteration $(itN):\t $((Eodd + Eeven) / 2)")
		end
	end
	
	return (l1, l2, g1, g2), Eiter
end

# ls : lambda, at side/ lc : lambda, at center (at figure in L10.2, page 3)
# gl : gamma, at left / gr : gamma, at right
function updatebond(ls, gl, lc, gr, expH, Hli; Nkeep=10)
	# l3prod : product of left 3 tensors
	# r2prod : product of right 2 tensors
	l3prod = (ls * gl) * lc
	r2prod = gr * ls
	totalprod = noprime((l3prod * r2prod) * expH)
	grls = commonind(gr, ls); lsgl = commonind(ls, gl)
	gllc = commonind(gl, lc); lcgr = commonind(lc, gr)

	(U, nlc, V), _ = svd(totalprod, grls, Hli; Nkeep, cutoff=1e-8)
	replacecommoninds!(U, nlc, gllc)
	replacecommoninds!(V, nlc, lcgr)

	lsinv_arr = inv(ls.arr)
	# prefix 'n' in variable name means 'new'
	ngl = U * LurTensor(lsinv_arr, grls, lsgl)
	ngr = V * LurTensor(lsinv_arr, lsgl, grls)
	return ngl, nlc, ngr
end

function getenergy(l2, g1, l1, g2, H_forodd, H_foreven, bonds)
	Cnull = contract_iTEBD(l2, g1, l1, g2, bonds)
	Codd = contract_iTEBD(l2, g1, l1, g2, H_forodd, 1, bonds)
	Ceven = contract_iTEBD(l2, g1, l1, g2, H_foreven, 2, bonds)
	return Codd / Cnull, Ceven / Cnull
end

contract_iTEBD(l2, g1, l1, g2, bonds) = 
	contract_iTEBD(l2, g1, l1, g2, nothing, -10, bonds)

function contract_iTEBD(l2, g1, l1, g2, lt, when, bonds)
	seq = [(l2 * g1) * l1, g2 * l2, g1 * l1]
	ldi = commonind(l2, g2)
	rdi = commonind(l1, g2)
	l1sz = size(l1.arr)[1]
	l2sz = size(l2.arr)[1]
	LT = LurTensor(Matrix(I, l2sz, l2sz), ldi, ldi')
	for i in 1:length(seq)
		t = seq[i]
		if i == 3
			tb = replaceind(t', rdi', rdi)
		else
			tb = t'
		end

		if i == when
			LT = ((LT * t) * tb) * lt
		elseif i == when + 1
			LT = (LT * t) * tb
		else
			sind = uniqueind(t, bonds)
			LT = (LT * t) * replaceind(tb, sind', sind)
		end
	end
	return value(LT)
end
