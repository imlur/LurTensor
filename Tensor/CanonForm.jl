using LinearAlgebra

canonform(M, id, ldi, rdi; kw...) = canonform(M, id, Index(ldi), Index(rdi); kw...)
function canonform(M::Vector{LurTensor}, id::Int, ldi::Index, rdi::Index; kw...)
	nt = length(M) # number of tensors
	if !(id in 0:nt)
		error("The 2nd input 'id' needs to be in a range (0:length(M))")
	elseif get_size(M[1], ldi) != 1
		error("The leftmost leg of M[1] should be of size 1")
	elseif get_size(M[end], rdi) != 1
		error("The rightmost leg of M[end] should be of size 1")
	end

	bond_inds = [ldi, [check_common_inds(M[i], M[i+1], "M[$(i)]", "M[$(i+1)]") for i=1:nt-1]..., rdi]
	site_inds = [uniqueind(M[i], bond_inds) for i=1:nt]
	dw = zeros(Float64, nt - 1)

	for i=1:id-1
		(U, S, Vd), d = svd(M[i], bond_inds[i], site_inds[i]; kw...)
		M[i] = replaceind(U, "svd,left", bond_inds[i+1])
		M[i+1] = replaceind((S * Vd) * M[i+1], "svd,left", bond_inds[i+1])
		dw[i] = d
	end

	for i=nt:-1:id+2
		(U, S, Vd), d = svd(M[i], bond_inds[i]; kw...)
		M[i] = replaceind(Vd, "svd,right", bond_inds[i])
		M[i-1] = replaceind((M[i-1] * U) * S, "svd,right", bond_inds[i])
		dw[i-1] = d
	end

	if id == 0
		(U, final_S, Vd), _ = svd(M[1], ldi)
		M[1] = replaceind(U, "svd,left", "svd,right") * Vd
	elseif id == nt
		(U, final_S, Vd), _ = svd(M[end], bond_inds[end-1], site_inds[end])
		M[end] = U * replaceind(Vd, "svd,right", "svd,left")
	else
		(U, S, Vd), d = svd(M[id] * M[id+1], bond_inds[id], site_inds[id]; kw...)
		M[id] = replaceind(U, "svd,left", bond_inds[id+1])
		M[id+1] = replaceind(Vd, "svd,right", bond_inds[id+1]')
		final_S = replaceinds(S, ["svd,left", "svd,right"], [bond_inds[id+1], bond_inds[id+1]'])
		dw[id] = d
	end

	return M, final_S, dw
end
