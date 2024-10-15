# For general case
# Indices should be prepared before this function
function updateleft(Cleft, B, X, A)
    check_tensors(Cleft, B, X, A)
    return ((Cleft * A) * X) * B
end

# If you want to make Cleft identity tensor
updateleft(B, Blidx, X::LurTensor, A, Alidx) = 
	updateleft(B, Index(Blidx), X, A, Index(Alidx))

function updateleft(B, Blidx::Index, X::LurTensor, A, Alidx::Index)
    check_tensors([Alidx, Blidx], B, X, A)
    return (B * X) * replaceind(A, Alidx, Blidx)
end

# If you want to make X identity tensor
updateleft(Cleft, B::LurTensor, Bsidx, A, Asidx) =
	updateleft(Cleft, B, Index(Bsidx), A, Index(Asidx))

function updateleft(Cleft, B::LurTensor, Bsidx::Index, A, Asidx::Index)
    check_tensors(Cleft, B, [Asidx, Bsidx], A)
    return (Cleft * B) * replaceind(A, Asidx, Bsidx)
end

# If you want to make both Cleft and X identity
updateleft(B, Blidx, Bsidx, A, Alidx, Asidx) =
	updateleft(B, Index(Blidx), Index(Bsidx), A, Index(Alidx), Index(Asidx))

function updateleft(B, Blidx::Index, Bsidx::Index, A, Alidx::Index, Asidx::Index)
    check_tensors([Alidx, Blidx], B, [Asidx, Bsidx], A)
    return B * replaceinds(A, Asidx=>Bsidx, Alidx=>Blidx)
end

function check_tensors(Cleft, B, X, A)
	check_common_inds(Cleft, A, "Cleft", "A")
	check_common_inds(Cleft, B, "Cleft", "B")
	check_common_inds(A, X, "A", "X")
	check_common_inds(B, X, "B", "X")
	check_common_inds(Cleft, X, Val(order(Cleft)), Val(order(X)))
end

check_common_inds(Cleft, X, ::Val{3}, ::Val{3}) = 
	check_common_inds(Cleft, X, "Cleft", "X")

check_common_inds(Cleft, X, ::Val{4}, ::Val{3}) = 
	check_common_inds(Cleft, X, "Cleft", "X")

check_common_inds(Cleft, X, ::Val{3}, ::Val{4}) = 
	check_common_inds(Cleft, X, "Cleft", "X")

check_common_inds(Cleft, X, ::Val{4}, ::Val{4}) = 
	check_common_inds(Cleft, X, "Cleft", "X")

check_common_inds(Cleft, X, a, b) = nothing

function check_common_inds(A, B, Aname::String, Bname::String)
	ci = commoninds(A, B)
    if length(ci) < 1
        error("There are no common indices between $(Aname) and $(Bname)")

    #elseif length(ci) > 1
    #    error("There are too many common indices between $(Aname) and $(Bname)")
    end
	return ci[1]
end
