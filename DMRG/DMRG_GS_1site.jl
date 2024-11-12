function DMRG_GS_1site!(M, Hs; Nkeep=30, Nsweep=4, Krylov=5, tol=1e-8)
	# Initialization, start from left-canonical form
	L = length(M)
	Eiter = zeros(L, 2 * Nsweep)
	Sv = zeros(L + 1)

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
		for itN=L:-1:1
			println(norm(M[itN]))
		end

		# left -> right
		for itN=1:L
		end
	end
	return M, 0, Eiter, Sv
end
