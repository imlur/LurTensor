tDMRG!(M, Hs, ldi, rdi; kw...) = tDMRG!(M, Hs, Index(ldi), Index(rdi); kw...)
# op : should have two Index, one of them has plev=0, the other has plev=1
# and two Index should have same tag
function tDMRG!(M, Hs, ldi::Index, rdi::Index; 
				Nkeep=20, dt=1/20, tmax=20, op=nothing)
	L = length(M)
	Nstep = Int(ceil(tmax / dt))

	# List of tags in M tensors
	bond_inds = [check_common_inds(M[i], M[i+1], 
				"M[$(i)]", "M[$(i+1)]") for i=1:L-1]
	bond_inds = [ldi, bond_inds..., rdi]
	site_inds = [uniqueind(M[i], bond_inds) for i=1:L]

	ts = dt * collect(1:Nstep)
	Ovals = zeros(ComplexF64, Nstep, L)
	EE = zeros(3*Nstep, L-1)
	dw = zeros(size(EE)...)

	println("tDMRG : Real-time evolution with local measurements\n")
	println("System size = $(L), Nkeep = $(Nkeep), 
			dt = $(dt), tmax = $(tmax), $(Nstep) steps")

	# get exponential of two-site gates
	expH = Vector{LurTensor}(undef, L-1)
	for i=1:L-1
		if isassigned(Hs, i)
			H = Hs[i]
			li, ri = commonind(H, M[i]), commonind(H, M[i+1])
			Htmp = permutedims(H, [li', ri', li, ri])
			Harr = Htmp.arr; D = prod(size(Harr)[1:2])
			Hmat = reshape(Harr, D, D)

			ttmp = i%2 == 1 ? dt/2 : dt
			evals, evecs = eigen(Hmat)
			expH_mat = evecs * Diagonal(exp.(-ttmp*im*evals)) * evecs'
			expH_arr = reshape(expH_mat, size(Harr)...)
			expH[i] = LurTensor(expH_arr, [li', ri', li, ri])
		end
	end

	M, _, _ = canonform(M, 0, ldi, rdi)

	# si : sweep index
	for si = 1:3*Nstep
		range = si % 3 == 2 ? (2:2:L-1) : (1:2:L-1)
		isright = si % 2 == 0
		EE[si, :], dw[si, :] = tDMRG_1sweep!(M, expH, range; Nkeep, isright)
		if si % 3 == 0 && !isnothing(op)
			Ovals[div(si, 3), :] = tDMRG_expVal(M, op, 
								  si % 2 == 0, ldi, rdi, site_inds)
		end
		if si % round(3 * Nstep / 10) == 0 || si == 3 * Nstep
			println("Sweep $(si) / $(3 * Nstep)")
		end
	end
	return ts, M, Ovals, EE, dw
end

function tDMRG_1sweep!(M, expH, range;Nkeep=20, isright)
	L = length(M)
	EE = zeros(L - 1)
	dw = zeros(L - 1)
	cutoff = 1e-8
	irange = isright ? (1:L-1) : (L-1:-1:1)

	for i in irange
		bondi = commonind(M[i], M[i+1])
		T = M[i] * M[i+1]
		if i in range
			T *= expH[i]
			prime!(T, -1; inds=commoninds(T, expH[i]))
		end
		(U, S, V), dw[i] = svd(T, commoninds(T, M[i]); Nkeep, cutoff)
		S = S / norm(S)
		Spart = diag(S)[diag(S) .> 0]; 
		EE[i] = sum(-(Spart.^2).*log.(Spart.^2)/log(2))

		M[i] = isright ? U : U * S
		M[i+1] = isright ? S * V : V
		replacecommoninds!(M[i], M[i+1], bondi)
	end
	lidx = isright ? L : 1 # last index
	M[lidx] = M[lidx] / norm(M[lidx])
	return EE, dw
end

function tDMRG_expVal(M, op, isleft, ldi, rdi, sinds)
	L = length(M)
	Ovals = zeros(ComplexF64, L)
	opi = ind(op; plev=0); op = permutedims(op, [opi, opi'])
	
	sdi = isleft ? rdi : ldi # start index
	MM = LurTensor(reshape([1], 1, 1), sdi, sdi')
	range = isleft ? (L:-1:1) : (1:L)

	for i in range
		op = LurTensor(op, sinds[i], sinds[i]')
		T2 = M[i] * op
		oval_tmp = MM * T2
		Ovals[i] = value(oval_tmp * 
			conj(prime(M[i]', -1; inds=uniqueind(M[i]', oval_tmp))))
		MM = MM * M[i] * conj(replaceind(M[i]', sinds[i]', sinds[i]))
	end
	return Ovals
end
