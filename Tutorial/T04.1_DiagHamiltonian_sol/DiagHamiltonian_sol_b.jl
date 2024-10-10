include("../../Tensor/LurTensor.jl")


# Initialize
N = 3
J = 1
Ss = Vector{LurTensor}(undef, N)


# unprimed Index : ket, primed Index : bra
stag = ["site,$(i)" for i=1:N] # site tag

let
	H = LurTensor(reshape([0], 1, 1), ["ldm", "ldm"], [0, 1])
	left_tag = "ldm"; left_dim = 1

	for i=1:N
		Anow = getidentity([left_dim, 2], left_tag, stag[i], "$(i)")
		H = updateleft(H, Anow', (stag[i], 1), Anow, (stag[i], 0))
		
		for i2=1:i-1
			S_dag = getlocalspace("Spin", 0.5, 'S', (stag[i], 0), (stag[i], 1), "S")
			H = H + J * updateleft(Ss[i2], Anow', S_dag, Anow)
		end

		for i2=1:i
			if i2 < i
				Ss[i2] = updateleft(Ss[i2], Anow', (stag[i], 1), 
											Anow, (stag[i], 0))
			else
				S = getlocalspace("Spin", 0.5, 'S', (stag[i], 1), (stag[i], 0), "S")
				Ss[i2] = updateleft(Anow', (left_tag, 1), S, Anow, (left_tag, 0))
			end
		end
		left_tag = "$(i)"; left_dim *= 2
	end
	show(stdout, H, showarr=true)
	#arr = Array{Float64, 2}(H, inds(H)...)
	println(round.(eigvals(H.arr), digits=2))
end

