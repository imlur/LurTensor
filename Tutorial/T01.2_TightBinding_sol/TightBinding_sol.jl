using LinearAlgebra

# This function evaluates eps for non-floating point values
function eps(t::Real)
	return Base.eps(Float64(t))
end

# This function evaluates eps for complex number 
function eps(t::Complex{<:Real})
	return max(eps(real(t)), eps(imag(t)))
end



function nonIntTB(t::Vector{T}) where T<:Number
	l = length(t) + 1
	H_1p::Matrix{T} = zeros(T, l, l)

	for i=1:length(t)
		H_1p[i, i+1] = conj(t[i]); H_1p[i+1, i] = t[i]
	end
	
	# Vector of eigenvalues, already sorted in ascending order
	eigvs = round.(eigvals(H_1p), digits=4)
	println(eigvs)
	# numerical precision noise
	noise = 10 * l * maximum(eps.(H_1p))
	eigvs = [abs(x) < noise ? 0 : x for x in eigvs]

	ground_energy = sum(eigvs .* (eigvs .< 0))
	zero_count = sum(eigvs .== 0)
	println("Ground energy:\t\t $(ground_energy)")
	println("Ground degeneracy:\t $(2^zero_count)")
end

println("For L = 10, t_l = 1")
nonIntTB([1 for i=1:9])
println()

println("For L = 11, t_l = 1")
nonIntTB([1 for i=1:10])
println()

println("For L = 11, t_l = e^il")
nonIntTB([exp(im*i) for i=1:10])
println()
