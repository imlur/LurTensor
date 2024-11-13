mpsnorm(mps, ldi, rdi) = mpsnorm(mps, Index(ldi), Index(rdi))
function mpsnorm(M::Vector{LurTensor}, ldi::Index, rdi::Index)
	nt = length(M)
	bond_inds = [ldi, [check_common_inds(M[i], M[i+1], "M[$(i)]", "M[$(i+1)]") for i=1:nt-1]..., rdi]
	site_inds = [uniqueind(M[i], bond_inds) for i=1:nt]
	t = M[1] * prime(M[1]', -1; tag=site_inds[1].tag)
	for i=2:nt
		t = t * M[i]
		t = t * prime(M[i]', -1; tag=site_inds[i].tag)
	end
	return value(t)
end

getenergy(M, Hs, ldi, rdi) = getenergy(M, Hs, Index(ldi), Index(rdi))
function getenergy(M::Vector{LurTensor}, Hs::Vector{LurTensor}, ldi::Index, rdi::Index)
	nt = length(M)
	bond_inds = [ldi, [check_common_inds(M[i], M[i+1], "M[$(i)]", "M[$(i+1)]") for i=1:nt-1]..., rdi]
	site_inds = [uniqueind(M[i], bond_inds) for i=1:nt]
	
	e = LurTensor(reshape([1], 1, 1), ldi, ldi')
	for i=1:nt
		e = updateleft(e, M[i]', Hs[i], M[i])
	end
	return value(e)
end