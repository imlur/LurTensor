include("../../Tensor/LurTensor.jl")

N = 11
t = exp.(im * (1:N-1))

stag = ["site,$(i)" for i=1:N] # site tag

let
	H = LurTensor(reshape([0], 1, 1), ["ldm", "ldm"], [0, 1])
	left_tag = "ldm"; left_dim = 1
	Fprev = LurTensor()

	for i=1:N
		F = getlocalspace("Fermion", 'F', (stag[i], 1), (stag[i], 0))
		Anow = getidentity([left_dim, 2], left_tag, stag[i], "$(i)")
		H = updateleft(H, Anow', (stag[i], 1), Anow, (stag[i], 0))

		if i > 1
			Z = getlocalspace("Fermion", 'Z', (stag[i], 1), (stag[i], 2))
			FF = getlocalspace("Fermion", 'F', (stag[i], 2), (stag[i], 0))
			ZFconj = hconj(Z * FF)
			Hhop = -t[i-1] * updateleft(Fprev, Anow', ZFconj, Anow)
			H = H + Hhop + hconj(Hhop)
		end

		Fprev = updateleft(Anow', (left_tag, 1), F, Anow, (left_tag, 0))
		left_tag = "$(i)"; left_dim *= 2
	end
	display(H)
	H_mat = H.arr
	Es = sort(eigvals(H_mat))
	println(round.(Es[1:10], digits=4))
end
