using LinearAlgebra

function lowesteigs(init, mft, args...; verbose=false, v0s=[], kw...)
	# Normalize first
	init = init / norm(init)
	T, V, nv = lanczos(init, mft, args...; v0s, kw...)
	nv0 = length(v0s)
	vals, vecs = eigen(T)
	val, vec = vals[1], vecs[:, 1]
	if verbose
		(dot(V[1].arr, V[nv+nv0].arr) > 0.1) && error("not orthogonal")
	end
	return sum([vec[i] * V[i+nv0] for i=1:min(nv, length(vec))]), val
end

# init : initial guess
# mft : get matrix * vector
# args : arguments of mft
# notations are from TN2022 cource lecture note L08.1
# can be used for both ordinary Matrix and LurTensor
# TODO: change code to use Matrix if T is vector
function lanczos(init::T, mft, args...; Krylov=5, tol=1e-8, v0s=[]) where T
	# d: diagonal, sd: sub-diagonal
	nv0 = length(v0s)
	d, sd = zeros(Krylov), zeros(Krylov - 1)
	V = Vector{T}(undef, nv0 + Krylov)
	V[1:nv0] = v0s

	# orthonormalize twice
	init = orthonormalize(init, V, nv0)
	init = orthonormalize(init, V, nv0)
	v = init / norm(init)
	u = v; V[nv0+1] = v
	#println("dot1 : ", dot(V[1], v))

	nv = 1
	for i=1:Krylov
		w = mft(v, args...)
		d[i] = dot(v, w)
		u = orthonormalize(w, V, nv0+i)
		u = orthonormalize(u, V, nv0+i)
		#println(norm(u))
		#println("dot2 : ", dot(V[1], v))
		nu = norm(u)
		(nu < tol) && break

		v = u / nu
		if i < Krylov
			sd[i] = nu
			V[nv0+i+1] = v
			nv += 1
		end
	end
	return SymTridiagonal(d, sd), V, nv
end

function orthonormalize(input, Vectors, n)
	(n == 0) && return input
	Vw = [dot(Vectors[j], input) for j=1:n]
	return input - sum([Vectors[j] * Vw[j] for j=1:n])
end
