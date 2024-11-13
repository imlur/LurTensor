using LinearAlgebra

function lowesteigs(init, mft, args...; Krylov=5, tol=1e-8)
	# Normalize first
	init = init / norm(init)
	T, V, nv = lanczos(init, mft, args...; Krylov, tol)
	vals, vecs = eigen(T)
	val, vec = vals[1], vecs[:, 1]
	return sum([vec[i] * V[i] for i=1:min(nv, length(vec))]), val
end

# init : initial guess
# mft : get matrix * vector
# args : arguments of mft
# notations are from TN2022 cource lecture note L08.1
# can be used for both ordinary Matrix and LurTensor
# TODO: change code to use Matrix if T is vector
# TODO: check for LurTensor case
function lanczos(init::T, mft, args...; Krylov=5, tol=1e-8) where T
	# d: diagonal, sd: sub-diagonal
	v = init; u = init
	d, sd = zeros(Krylov), zeros(Krylov - 1)
	V = Vector{T}(undef, Krylov)
	V[1] = v

	nv = 1
	for i=1:Krylov
		#display(v)
		w = mft(v, args...)
		#display(w)
		#display(matchlegs(v, w))
		# sum(elementwise product of complex conjugate of v and w)
		# = inner product
		d[i] = dot(v, w)
		Vw = [dot(V[j], w) for j=1:i]
		u = w - sum([V[j] * Vw[j] for j=1:i])
		nu = norm(u)
		(nu < tol) && break

		v = u / nu
		if i < Krylov
			sd[i] = nu
			V[i+1] = v
			nv += 1
		end
	end
	return SymTridiagonal(d, sd), V, nv
end
