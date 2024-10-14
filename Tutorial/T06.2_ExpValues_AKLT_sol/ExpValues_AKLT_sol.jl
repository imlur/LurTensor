include("../../Tensor/LurTensor.jl")

# Generate AKLT state 
AKLT = zeros(2, 2, 3)

# Local spin S_z = +1
AKLT[1, 2, 1] = sqrt(2/3)
# Local spin S_z = 0
AKLT[1, 1, 2] = -1 / sqrt(3)
AKLT[2, 2, 2] = 1 / sqrt(3)
# Local spin S_z = -1
AKLT[2, 1, 3] = -sqrt(2/3)

L = 50
site_tags = ["site,$(i)" for i=1:L]
bond_tags = ["ld", ["$(i)~$(i+1)" for i=1:L-1]..., "rd"]

# Get left-canonical form of AKLT state with specific boundary condition
function get_normalized_AKLT(L, alpha, beta)
	M = Vector{LurTensor}(undef, L)
	for i=1:L
		T = LurTensor(AKLT, bond_tags[i], bond_tags[i+1], site_tags[i])
		if i == 1
			M[i] = T[alpha:alpha, :, :]
		elseif i == L
			M[i] = T[:, beta:beta, :]
		else
			M[i] = T
		end
	end
	M, _, _ = canonform(M, L, "ld", "rd")
	return M
end

# M : Left-normalized AKLT state
# L : system size
# szsites : site indices of site with Sz operator
function contract_AKLT(M, L, szsites...)
	# Initialize
	s = min(szsites...) # start site
	if s > 1
		T = LurTensor([1. 0.; 0. 1.], (bond_tags[s], 1), (bond_tags[s], 0))
	else
		T = LurTensor(reshape([1], 1, 1), ["ld", "ld"], [0, 1])
	end

	for i=s:L
		# site with Sz operator
		if i in szsites
			S = getlocalspace("Spin", 1, 'S', (site_tags[i], 1), (site_tags[i], 0), "_")
			Sz = S[:, :, 2] 
			T = updateleft(T, M[i]', Sz, M[i])
		else
			T = updateleft(T, M[i]', (site_tags[i], 1), M[i], (site_tags[i], 0))
		end
	end
	return T[1]
end

function exact_m(a, b, i, L)
	deno = 1 + (-1)^(a + b) * (-1 / 3)^L
	numer = (-1 / 3)^i - (-1)^(a + b) * (-1 / 3)^(L - i + 1)
	return 2 * (-1)^a * numer / deno
end

function exact_corr(a, b, i, L)
	deno = 1 + (-1)^(a + b) * (-1 / 3)^L
	numer = -4/9 - 4 * (-1)^(a + b) * (-1 / 3)^L
	return numer / deno
end


let
	# Calculated from tensor network
	Ms_tn = zeros(2, 2, L)
	Corrs_tn = zeros(2, 2, L-1)

	# Exact value
	Ms_exact = zeros(2, 2, L)
	Corrs_exact = zeros(2, 2, L-1)

	for a=1:2
		for b=1:2
			M = get_normalized_AKLT(L, a, b)
			for i=1:L
				Ms_tn[a, b, i] = contract_AKLT(M, L, i)
				Ms_exact[a, b, i] = exact_m(a, b, i, L)
			end

			for i=1:L-1
				Corrs_tn[a, b, i] = contract_AKLT(M, L, i, i+1)
				Corrs_exact[a, b, i] = exact_corr(a, b, i, L)
			end
		end
	end

	println("\n\n\nExpected value of magnetization from tensor network : ")
	show(stdout, "text/plain", Ms_tn)
	println("\n\n\nExpected value of magnetization from analytic result: ")
	show(stdout, "text/plain", Ms_exact)
	println("\n\n")

	# Difference between tensor network value and analytic result
	# is about numerical noise
	println(norm(Ms_tn - Ms_exact))
	println(norm(Corrs_tn - Corrs_exact))
end
