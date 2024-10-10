function wstate(N::Int)
	A = zeros([2 for _=1:N]...)
	A[[2^(i-1)+1 for i=1:N]] .= 1 / sqrt(N)
	return A
end

N = 5
A = wstate(N)
println(A[2, 1, 1, 1, 1] * sqrt(N))
println(A[1, 2, 1, 1, 1] * sqrt(N))
println(A[1, 1, 2, 1, 1] * sqrt(N))
println(A[1, 1, 1, 2, 1] * sqrt(N))
println(A[1, 1, 1, 1, 2] * sqrt(N))
