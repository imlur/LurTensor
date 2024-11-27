include("../../Tensor/LurTensor.jl")
using SpecialFunctions
using Plots

let
	J = -1; L = 50;
	stags = ["site,$(i)" for i=1:L] # Site tag
	btags = ["ld", ["$(i)~$(i+1)" for i=1:L-1]..., "rd"] # Bond leg tag for state

	Sarr = getlocalspace("Spin", 0.5, 'S')
	Sarr_xy = Sarr[:, :, [1, 3]]
	S1 = LurTensor(Sarr_xy, ("sl", 1), ("sl", 0), "S")
	S2 = conj(LurTensor(Sarr_xy, ("sr", 0), ("sr", 1), "S"))
	H = J * S1 * S2
	println(norm(H - swapprime(H, 0, 1)))
	Hs = Vector{LurTensor}(undef, L-1)
	for i=1:L-1
		Hs[i] = replacetags(replacetags(H, "sl", stags[i]), "sr", stags[i+1])
	end

	Nkeep, dt, tmax = 20, 1/20, 20
	Sz = LurTensor(Sarr[:, :, 2], ("op", 1), ("op", 0))
	M = Vector{LurTensor}(undef, L)
	linit = reshape([1, 0], 2, 1, 1); rinit = reshape([0, 1], 2, 1, 1)
	for i=1:L
		M[i] = LurTensor(i <= (L/2) ? linit : rinit, stags[i], btags[i], btags[i+1])
	end
	ts, M, Ovals, EE, dw = tDMRG!(M, Hs, "ld", "rd"; Nkeep, dt, tmax, op=Sz)
	println(norm(imag(Ovals)))
	
	display(heatmap(real(Ovals[end:-1:1, :])))
	display(heatmap(EE[end:-1:1, :]))

	fvals = zeros(length(ts), L - 1)
	for it=1:L-1
		for t=1:length(ts)
			fvals[t, it] = besselj(it-div(L,2), ts[t])^2
		end
	end
	fvals = -0.5 * fvals

	Oexact = zeros(length(ts), div(L, 2))
	for it=1:div(L, 2)
		for t=1:length(ts)
			Oexact[t, it] = sum(fvals[t, (div(L, 2)-it+1):(it+div(L, 2)-1)])
		end
	end

	errors = zeros(length(ts))
	for t=1:length(ts)
		errors[t] = max(abs.(Ovals[t, div(L, 2)+1:end] - Oexact[t, :])...)
	end
	display(plot(errors, yscale=:log10))
end
