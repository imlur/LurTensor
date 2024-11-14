# GS : ground state MPS
# This code can be generalized to nth excited state
# if the first n states are provided
function DMRG_ES_1site!(M, Hs, GS; Nkeep=30, Nsweep=4, Krylov=5, tol=1e-8)
	# Initialization, start from left-canonical form
	L = length(M)
	Eiter = zeros(L, 2 * Nsweep)
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
	
	Olr = Vector{LurTensor}(undef, L + 2)
	Olr[1] = LurTensor(reshape([1], 1, 1), ldi, ldi')
	Olr[end] = LurTensor(reshape([1], 1, 1), rdi, rdi')

	Hlr = Vector{LurTensor}(undef, L + 2)
	Hlr[1] = LurTensor(reshape([1], 1, 1), ldi, ldi')
	Hlr[end] = LurTensor(reshape([1], 1, 1), rdi, rdi')
	for i=1:L
		Hlr[i+1] = updateleft(Hlr[i], M[i]', Hs[i], M[i])
		Olr[i+1] = updateleft(Olr[i], M[i]', site_inds[i]', 
									  GS[i], site_inds[i])
	end

	for itS=1:Nsweep
		# right -> left
		for itN=L:-1:1
			Aorth = noprime((Olr[itN] * GS[itN]) * Olr[itN+2])
			t, E = lowesteigs(M[itN], mft_1GS, 
				Hlr[itN], Hs[itN], Hlr[itN+2]; v0s=[Aorth], tol=1e-4)
			#println(t * Aorth)

			Eiter[L+1-itN, 2*itS-1] = E
			(U, S, Vd), _ = svd(t, bond_inds[itN])
			Sv[itN] = S
			M[itN] = Vd
			if itN > 1
				M[itN-1] = (M[itN-1] * U) * S
				replacetags!(M[itN], "svd,right", bond_inds[itN].tag)
				replacetags!(M[itN-1], "svd,right", bond_inds[itN].tag)
			else
				M[1] = replacetags(U, "left", "right") * Vd
			end

			Hlr[itN+1] = updateleft(Hlr[itN+2], M[itN]', Hs[itN], M[itN])
			Olr[itN+1] = updateleft(Olr[itN+2], M[itN]', site_inds[itN]', 
										  GS[itN], site_inds[itN])
			#println(innerprod(M, GS, "ld", "rd"))
		end
		#println(getenergy(M, Hs, "ld", "rd"))

		# left -> right
		for itN=1:L
			Aorth = noprime((Olr[itN] * GS[itN]) * Olr[itN+2])
			# tol=1e-4 is due to precision issue
			t, E = lowesteigs(M[itN], mft_1GS, 
				Hlr[itN], Hs[itN], Hlr[itN+2]; v0s=[Aorth], tol=1e-4)
			Eiter[itN, 2*itS] = E
			(U, S, Vd), _ = svd(t, bond_inds[itN], site_inds[itN])
			Sv[itN+1] = S
			M[itN] = U
			if itN < L
				M[itN+1] = (M[itN+1] * Vd) * S
				replacetags!(M[itN], "svd,left", bond_inds[itN+1].tag)
				replacetags!(M[itN+1], "svd,left", bond_inds[itN+1].tag)
			else
				M[L] = replacetags(Vd, "right", "left") * U 
			end

			Hlr[itN+1] = updateleft(Hlr[itN], M[itN]', Hs[itN], M[itN])
			Olr[itN+1] = updateleft(Olr[itN], M[itN]', site_inds[itN]', 
										  GS[itN], site_inds[itN])
		end
		println(getenergy(M, Hs, "ld", "rd"))
	end
	return getenergy(M, Hs, "ld", "rd"), Eiter, Sv
end



function mft_1GS(lt, Hleft, Hcen, Hright) 
	#display(lt)
	#display(Hleft)
	#display(Hcen)
	#display(Hright)
	noprime(((lt * Hleft) * Hcen) * Hright)
end
