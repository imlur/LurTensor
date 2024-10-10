using LinearAlgebra

Val_dict = Dict("Spin" => Val(1), "Fermion" => Val(2), "FermionS" => Val(3))
getlocalspace(space, op::Char, inds...) = getlocalspace(space, -1, op, inds...)
getlocalspace(space, aux::Number, op::Char, inds...) = LurTensor(get_arr(Val_dict[space], aux, Val(op)), inds...)

check_s(S) = round(S*2) != S*2 ? error("invalid spin value") : nothing

get_arr(::Any, ::Any, ::Any) = error("Not supported operator")
get_arr(::Val{1}, S, ::Val{'I'}) = (check_s(S); d = round(2*S + 1); Matrix(I, d, d))
get_arr(::Val{2}, aux, ::Val{'I'}) = Matrix(I, 2, 2)
get_arr(::Val{3}, aux, ::Val{'I'}) = Matrix(I, 4, 4)

get_arr(::Val{2}, aux, ::Val{'F'}) = [0. 1.; 0. 0.]
get_arr(::Val{2}, aux, ::Val{'Z'}) = [1. 0.; 0. -1.]
get_arr(::Val{3}, aux, ::Val{'Z'}) = diagm([1. -1. -1. 1.])

function get_arr(::Val{3}, aux, ::Val{'F'})
    arr = zeros(4, 4, 2)
    arr[1, 2, 1] = arr[1, 3, 2] = arr[2, 4, 2] = 1; arr[3, 4, 1] = -1
    return arr
end

function get_arr(::Val{3}, aux, ::Val{'S'})
	arr = zeros(4, 4, 3)
	arr[2, 3, 1] = arr[3, 2, 3] = 1 / sqrt(2)
	arr[2, 2, 2] = 0.5; arr[3, 3, 2] = -0.5
    return arr
end

function get_arr(::Val{1}, S, ::Val{'S'})
    dim = Int(round(2*S+1)); arr = zeros(dim, dim, 3)
    m_range = S-1:-1:-S
    for i=1:dim-1
        m = m_range[i]
        arr[i, i+1, 1] = arr[i+1, i, 3] = sqrt((S-m)*(S+m+1)/2)
        arr[i+1, i+1, 2] = m
    end
    arr[1, 1, 2] = S
    return arr
end
