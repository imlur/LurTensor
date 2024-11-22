using KrylovKit

function ortho_orus(lambda, gamma, leftind)
	# phyinds : List of physical indices 
	# rightind : right-side leg of gamma tensor
	phyinds = uniqueinds(gamma, lambda)
	rightind = uniqueind(gamma, leftind, phyinds)
	tempind = Index("__temp__", 7373)

	M = replaceind(gamma, leftind, tempind) * lambda
	XR = ortho_orus_vec(M, leftind, tempind)

	M = lambda * replaceind(gamma, rightind, tempind)
	XL = ortho_orus_vec(M, rightind, tempind)

	(U, S, V), _ = svd(XL * lambda * XR, rightind''; cutoff=1e-8)
	replacecommoninds!(U, S, rightind')
	replacecommoninds!(S, V, leftind')
	VXRi = V * inv(XR); XLiU = inv(XL) * U
	ngamma = VXRi * gamma * XLiU
	return noprime(S), noprime(ngamma)
end

# cti : index where contraction bewteen V_R(V_L) and transfer matrix occurs
# tmpi : index at opposite side of trnasfer matrix
function ortho_orus_vec(M, cti, tmpi)
	T = M * replaceinds(M, [cti, tmpi], [cti', tmpi'])
	T = permutedims(T, [tmpi, tmpi', cti, cti'])
	D = size(T)[1]

	Tmat = reshape(T.arr, D^2, D^2)
	# Get one dominant eigenvalue / eigenvector
	_, vecs, _ = KrylovKit.eigsolve(Tmat)
	(norm(imag(vecs)) < 1e-8) && (vecs = real(vecs))
	Vmat = reshape(vecs[1], D, D)
	Vmat /= sign(tr(Vmat))
	VLT = LurTensor((Vmat + Vmat') / 2, cti, cti')
	
	(W, D, Wd), _ = eigen(VLT, cti')
	replacecommoninds!(W, D, cti'')
	return sqrt(D) * Wd
end
