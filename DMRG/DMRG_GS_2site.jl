function DMRG_GS_2site!(M, Hs; Nkeep=30, Nsweep=4, Krylov=5, tol=1e-8)
	# Initialization, start from left-canonical form
	L = length(M)
	Eiter = zeros(L-1, 2 * Nsweep)
	Sv = Vector{LurTensor}(undef, L + 1)

	# List of tags in M tensors
	site_inds = [check_common_inds(M[i], Hs[i], 
				"M[$(i)]", "Hs[$(i)]") for i=1:L]
	bond_inds = [check_common_inds(M[i], M[i+1], 
				"M[$(i)]", "M[$(i+1)]") for i=1:L-1]
	ldi = uniqueind(M[1], vcat(site_inds, bond_inds))
	rdi = uniqueind(M[L], vcat(site_inds, bond_inds))
	bond_inds = [ldi, bond_inds..., rdi]

	# left-canonical
	M, _, _ = canonform(M, L, ldi, rdi)
	
	Hlr = Vector{LurTensor}(undef, L + 2)
	Hlr[1] = LurTensor(reshape([1], 1, 1), ldi, ldi')
	Hlr[end] = LurTensor(reshape([1], 1, 1), rdi, rdi')
	for i=1:L
		Hlr[i+1] = updateleft(Hlr[i], M[i]', Hs[i], M[i])
	end

	for itS=1:Nsweep
		# right -> left
		for itN=L-1:-1:1
			Aold = M[itN] * M[itN+1]
			Anew, E = lowesteigs(Aold, mft_2GS, 
				Hlr[itN], Hs[itN], Hs[itN+1], Hlr[itN+3]; 
				Krylov=Krylov, tol=tol)
			Eiter[L-itN, 2*itS-1] = E

			(U, S, Vd), _ = svd(Anew, commoninds(Anew, M[itN]);
								Nkeep=Nkeep, cutoff=1e-8)
			M[itN] = replacetags(U * S, "svd,right", bond_inds[itN+1].tag)
			M[itN+1] = replacetags(Vd, "svd,right", bond_inds[itN+1].tag)
			Sv[itN+1] = S

			Hlr[itN+2] = updateleft(Hlr[itN+3], M[itN+1]', Hs[itN+1], M[itN+1])
		end
		println(getenergy(M, Hs, "ld", "rd"))

		# left -> right
		for itN=1:L-1
			Aold = M[itN] * M[itN+1]
			Anew, E = lowesteigs(Aold, mft_2GS, 
				Hlr[itN], Hs[itN], Hs[itN+1], Hlr[itN+3];
				Krylov=Krylov, tol=tol)
			Eiter[itN, 2*itS] = E

			(U, S, Vd), _ = svd(Anew, commoninds(Anew, M[itN]);
								Nkeep=Nkeep, cutoff=1e-8)
			M[itN] = replacetags(U, "svd,left", bond_inds[itN+1].tag)
			M[itN+1] = replacetags(S * Vd, "svd,left", bond_inds[itN+1].tag)
			Sv[itN+1] = S

			Hlr[itN+1] = updateleft(Hlr[itN], M[itN]', Hs[itN], M[itN])
		end
		println(getenergy(M, Hs, "ld", "rd"))
	end
	return getenergy(M, Hs, "ld", "rd"), Eiter, Sv
end

mft_2GS(lt, Hleft, Hcen1, Hcen2, Hright) =
	noprime((((lt * Hleft) * Hcen1) * Hcen2) * Hright)
