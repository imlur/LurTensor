include("../../Tensor/LurTensor.jl")
using Random


A = LurTensor(rand(4, 5, 3), "A", "B", "C")
B = LurTensor(rand(6, 4, 5), "D", "A", "B")
# Contract legs with same tag. 
C = A * B

println(C)

# If you want to permute dimension after contraction, use permutedims function
Cp = permutedims(C, [2, 1])
println(Cp)

# Unlike matlab, last trailling singleton dimensions are not removed
D = LurTensor(rand(3, 4, 1, 1, 1), "Z", "X", "C", "V", "B")
println(size(D))
println(D)
