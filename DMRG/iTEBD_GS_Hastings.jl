iTEBD_GS_Hastings(l1, l2, b1, b2, H, Hli, Hri, taus; kw...) =
	iTEBD_GS_Hastings(l1, l2, b1, b2, H, Index(Hli), Index(Hri), taus; kw...)

# Indices of H should be (Hli, Hli', Hri, Hri') (order does not matter)
function iTEBD_GS_Hastings(l1, l2, b1, b2, H, 
						 Hli::Index, Hri::Index, taus; Nkeep=10)
	Nstep = length(taus)
	Eiter = zeros(Nstep, 2, 2)
	l1 = LurTensor(Diagonal(l1.arr), l1.inds)
	l2 = LurTensor(Diagonal(l2.arr), l2.inds)
	nl1 = norm(l1)
	nl2 = norm(l2)
	l1 = l1 / nl1; b1 = b1 / nl1
	l2 = l2 / nl2; b2 = b2 / nl2
	
	l1g2 = check_common_inds(l1, b2, "l1", "b2")
	l2g1 = check_common_inds(l2, b1, "l2", "b1")
	l1g1 = uniqueind(l1, b1)
	l2g2 = uniqueind(l2, b2)
	bonds = [l1g1, l2g1, l1g2, l2g2]

	# odd site index, even site index
	osi = uniqueind(b1, bonds)
	esi = uniqueind(b2, bonds)
	
	# Reshape to exponentiate
	Hinds = [Hli', Hri', Hli, Hri]
	H = permutedims(H, Hinds)
	Hsz = size(H); rdim = Hsz[1] * Hsz[2]
	Hmat = reshape(H.arr, rdim, rdim)
	evals, evecs = eigen((Hmat + Hmat') / 2)
	# odd bonds 
	Hoind = [osi', esi', osi, esi]
	Heind = [esi', osi', esi, osi]
	H_forodd = LurTensor(H.arr, Hoind)
	H_foreven = LurTensor(H.arr, Heind)

	for itN=1:Nstep
		tau = taus[itN]
		expH_arr = evecs * Diagonal(exp.(-tau * evals)) * evecs'
		expH_arr = reshape(expH_arr, Hsz...)

		expH_forodd = LurTensor(expH_arr, Hoind)
		expH_foreven = LurTensor(expH_arr, Heind)

		b1, b2, l1 = updatebond_hast(l2, b1, b2, 
			expH_forodd, osi, l1g1, l1g2; Nkeep)
		Eodd, Eeven = getenergy_hast(l2, b1, l1, b2, 
			H_forodd, H_foreven, bonds)
		Eiter[itN, 1, :] = [Eodd, Eeven]

		b2, b1, l2 = updatebond_hast(l1, b2, b1, 
			expH_foreven, esi, l2g2, l2g1; Nkeep)
		Eodd, Eeven = getenergy_hast(l2, b1, l1, b2, 
			H_forodd, H_foreven, bonds)
		Eiter[itN, 2, :] = [Eodd, Eeven]

		if itN % 100 == 0
			println("Iteration $(itN):\t $((Eodd + Eeven) / 2)")
		end
	end
	return (l1, l2, b1, b2), Eiter
end

# ll : left l tensor, bl : b tensor at left side, br : b tensor at right side
function updatebond_hast(ll, bl, br, expH, Hli, clidx, cridx; Nkeep=10)
	llbl = commonind(ll, bl)
	ll = replaceind(ll, llbl, llbl')
	bl = replaceind(bl, llbl, llbl')

	phi = (bl * br) * expH
	llbr = commonind(phi, br)
	leftind = uniqueind(ll, bl)
	(U, nlc, nbr), _ = svd(noprime(ll * phi), leftind, Hli; Nkeep, cutoff=1e-8)
	replacecommoninds!(U, nlc, clidx)
	replacecommoninds!(nlc, nbr, cridx)
	
	nlcnorm = norm(nlc)
	phi /= nlcnorm; nlc /= nlcnorm
	nbl = phi * replaceind(nbr', llbr', llbr)
	return noprime(nbl), nbr, nlc
end

function getenergy_hast(l2::LurTensor, b1::LurTensor, 
						l1::LurTensor, b2::LurTensor, 
						H_forodd::LurTensor, 
						H_foreven::LurTensor, 
						bonds::Vector{Index})
	#l1inv = LurTensor(inv(l1.arr), l1.inds)
	#l2inv = LurTensor(inv(l2.arr), l2.inds)
	#g1 = b1 * l1inv; g2 = b2 * l2inv
	Cnull = contract_iTEBDH(l2, b1, l1, b2, bonds)
	Codd = contract_iTEBDH(l2, b1, l1, b2, H_forodd, 1, bonds)
	Ceven = contract_iTEBDH(l2, b1, l1, b2, H_foreven, 2, bonds)
	return Codd / Cnull, Ceven / Cnull
end

contract_iTEBDH(l2, b1, l1, b2, bonds) = 
	contract_iTEBDH(l2, b1, l1, b2, nothing, -10, bonds)

function contract_iTEBDH(l2, b1, l1, b2, lt, when, bonds)
	seq = [l2 * b1, b2, b1]
	ldi = uniqueind(l2, b2)
	rdi = commonind(l1, b2)
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
