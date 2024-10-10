using LinearAlgebra

getidentity(dims::Int...) = getidentity(collect(dims))
getidentity(dims::Vector{Int}) = getidentity(dims, [["id_in$(i)" for i=1:length(dims)]..., "id_out"])
getidentity(dims::Vector{Int}, inds...) = LurTensor(getidarr(dims), inds...)
getidarr(dims) = (pr = prod(dims); reshape(Matrix(I, pr, pr), dims..., pr))
