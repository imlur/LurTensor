using LinearAlgebra

struct Index
	tag::String
	plev::Int
end

Base.:(==)(x::Index, y::Index) = sort(x.tag) == sort(y.tag) && x.plev == y.plev
Base.hash(i::Index) = hash((sort(i.tag), i.plev))
Base.adjoint(i::Index) = Index(i.tag, i.plev + 1)
Index(ind::Index) = Index(ind.tag, ind.plev)
Index(t::Tuple{String, Int}) = Index(t[1], t[2])
Index(s::String) = Index(s, 0)

to_indvec(tps::Vector{Tuple{String, Int}}) = [Index(t) for t in tps]
to_indvec(ks::Vector{String}, ps::Vector{Int}) = to_indvec(collect(zip(ks, ps)))
to_indvec(ks::Vector{String}) = to_indvec(ks, [0 for _=1:length(ks)])
to_indvec(inds::Index...) = collect(inds)
to_indvec(ks...) = [Index(k) for k in ks]

struct LurTensor{T<:Number, N} <: AbstractArray{T, N}
	arr::AbstractArray{T, N}
	inds::Vector{Index}

	function LurTensor(arr::AbstractArray{T, N}, inds::Vector{Index}) where {T, N}
		if length(inds) != N
			error("Dimension of array is not equal to number of Index objects")
		elseif 0 in map(x -> length(x.tag), inds)
			error("Some of tags are empty string")
		end
		new{T, N}(arr, inds)
	end
end

LurTensor(arr::AbstractArray{T, N}, linds...) where {T, N} = 
	LurTensor(arr, to_indvec(linds...))
LurTensor(x::Number) = LurTensor([x], ["Null"])
LurTensor() = LurTensor(0)

function showDim(io::IO, LT::LurTensor{T, N}) where {T, N}
	sz = size(LT)
	println("\t\tSize\tplev\ttag")
	for d=1:N	
		println("Dim $(d) : \t$(sz[d]) \t$(LT.inds[d].plev) \t$(LT.inds[d].tag)")
	end
end

function Base.show(io::IO, LT::LurTensor; showarr=false)
	showDim(io, LT)
	if showarr
		println()
		show(io, "text/plain", LT)
	end
	println("\n\n")
end

Base.showarg(io::IO, LT::LurTensor, toplevel) = showDim(io, LT)

Base.IndexStyle(::Type{<:LurTensor}) = IndexLinear()
LurTensor(arr::AbstractArray{T, N}, tags) where {T, N} = LurTensor(arr, tags, [0 for _=1:N])

function LurTensor(LT::LurTensor{T, N}, I...) where {T, N}
	new_arr = getindex(LT.arr, I...)
	new_inds = [LT.inds[i] for i=1:N if !(I[i] isa Integer)]
	LurTensor(new_arr, new_inds)
end

Base.similar(LT::LurTensor{T, N}) where {N, T} = LurTensor(zeros(T, size(LT)...), LT.tags, LT.plevs)

order(LT::LurTensor) = length(size(LT))
order(v::Vector{Index}) = length(v)
Base.size(LT::LurTensor) = size(LT.arr)
Base.size(LT::LurTensor, dim::Int) = size(LT.arr, dim)

Base.getindex(LT::LurTensor, I...) = LurTensor(LT, I...)
Base.getindex(LT::LurTensor, i1::Union{Integer, CartesianIndex}, I::Union{Integer, CartesianIndex}...) = getindex(LT.arr, i1, I...)
Base.getindex(LT::LurTensor, c::Colon) = getindex(LT.arr, c)
Base.getindex(LT::LurTensor, I::AbstractUnitRange{<:Integer}) = getindex(LT.arr, I)
Base.setindex!(LT::LurTensor{T, N}, v, I...) where {T, N} = setindex!(LT.arr, v, I...)
Base.setindex!(LT::LurTensor{T, N}, v, i::Int) where {T, N} = setindex!(LT.arr, v, i)
Base.setindex!(LT::LurTensor{T, N}, v, i1::Int, i2::Int, I::Int...) where {T, N} = setindex!(LT.arr, v, i1, i2, I...)
Base.setindex!(LT::LurTensor, v, c::Colon) = setindex!(LT.arr, v, c)
Base.setindex!(LT::LurTensor, v, I::AbstractUnitRange{<:Integer}) = setindex!(LT.arr,v, I)


# Broadcast setting
Base.showarg(io::IO, A::LurTensor, toplevel) = print(io, typeof(A))
Base.BroadcastStyle(::Type{<:LurTensor}) = Broadcast.ArrayStyle{LurTensor}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{LurTensor}}, ::Type{ElType}) where ElType
	A = find_aac(bc)
	LurTensor(similar(Array{ElType}, axes(bc)), A.inds)
end

find_aac(bc::Base.Broadcast.Broadcasted) = find_aac(bc.args)
find_aac(args::Tuple) = find_aac(find_aac(args[1]), Base.tail(args))
find_aac(x) = x
find_aac(::Tuple{}) = nothing
find_aac(a::LurTensor, rest) = a
find_aac(::Any, rest) = find_aac(rest)

# Copy LurTensor object
Base.copy(LT::LurTensor) = LurTensor(Base.copy(LT.arr), Base.copy(LT.inds))
shallow_copy(LT::LurTensor) = LurTensor(LT.arr, Base.copy(LT.inds))

# Add condition to modify functions below
cond_kwargs = (tag=nothing, plev=nothing)

# Condition Function
function cd(kw)
	for k in keys(kw)
		if !(k in (:tag, :plev)) error("Keyword argument $(k) is not supported") end
	end
	t = get(kw, :tag, nothing); p = get(kw, :plev, nothing)
	(ind::Index) -> check(ind.tag, t) && check(ind.plev, p)
end

check(tag::String, str::String) = all(x -> x in split(tag, ','), split(str, ','))
check(plev::Int, i::Int) = plev == i
check(::Any, ::Nothing) = true

# Change tags
function remove_tag(ind::Index, str; check_empty=true)
	tag_split = split(ind.tag, ',')
	for s in split(str, ',') 
		filter!(e->e!=s, tag_split)
	end
	if check_empty && length(tag_split) == 0
		error("Empty tag is not allowed")
	end
	return Index(join(tag_split, ','), ind.plev)
end

function remove_dup(str::String)
	splitted = split(str, ','); result = Vector{String}()
	for a in splitted
		if a in result
			push!(result, a)
		end
	end
	return join(splitted, ',')
end

set_tag(ind::Index, str) = set_tag(ind, str, Val(length(str)))
set_tag(ind::Index, str, ::Val{0}) = error("Empty tag is not allowed")
set_tag(ind::Index, str, ::Any) = Index(remove_dup(str), ind.plev)

function add_tag(ind::Index, str::String)
	tag = ind.tag; tag_split = split(tag, ',')
	for s in split(str, ',')
		if !(s in tag_split)
			tag *= ',' * s 
		end
	end
	return Index(tag, ind.plev)
end

replace_tag(ind::Index, s1, s2) = replace_tag(ind, s1, s2, Val(check(ind.tag, s1)))
replace_tag(ind::Index, s1, s2, ::Val{true}) = add_tag(remove_tag(ind, s1; check_empty=false), s2)
replace_tag(ind::Index, s1, s2, ::Val{false}) = ind

swap_tag(ind::Index, s1, s2) = swap_tag(ind, s1, s2, Val(check(ind.tag, s1)), Val(check(ind.tag, s2)))
swap_tag(ind::Index, s1, s2, ::Val{true}, ::Val{false}) = replace_tag(ind, s1, s2, Val(true))
swap_tag(ind::Index, s1, s2, ::Val{false}, ::Val{true}) = replace_tag(ind, s2, s1, Val(true))
swap_tag(ind::Index, s1, s2, ::Any, ::Any) = ind

map_prime(ind, plold, plnew) = Index(ind.tag, (ind.plev == plold) ? plnew : ind.plev)
swap_prime(ind, pl1, pl2) = Index(ind.tag, (ind.plev == pl1) ? pl2 : (ind.plev == pl2) ? pl1 : ind.plev)


# Modify tags of LurTensor object (Inplace version)
addtags!(LT::LurTensor, str; kw...) = modify!(LT, add_tag, cd(kw), str)
removetags!(LT::LurTensor, str; kw...) = modify!(LT, remove_tag, cd(kw), str)
settags!(LT::LurTensor, str; kw...) = modify!(LT, set_tag, cd(kw), str)
replacetags!(LT::LurTensor, s1, s2; kw...) = modify!(LT, replace_tag, cd(kw), s1, s2)
swaptags!(LT::LurTensor, s1, s2; kw...) = modify!(LT, swap_tag, cd(kw), s1, s2)

# Modify tags of LurTensor object (Copy version)
removetags(LT::LurTensor, str; kw...) = modify(removetags!, LT, str; kw...)
addtags(LT::LurTensor, str; kw...) = modify(addtags!, LT, str; kw)
settags(LT::LurTensor, str; kw...) = modify(settags!, LT, str; kw...)
replacetags(LT::LurTensor, str1, str2; kw...) = modify(replacetags!, LT, str1, str2; kw...)
swaptags(LT::LurTensor, str1, str2; kw...) = modify(swaptags!, LT, str1, str2; kw...)

# Modify plevs of LurTensor object (Inplace version)
prime!(LT::LurTensor, plinc::Int = 1; kw...) = modify!(LT, (ind, pl) -> Index(ind.tag, max(0, ind.plev + pl)), cd(kw), plinc)
setprime!(LT::LurTensor, plev::Int; kw...) = modify!(LT, (ind, pl) -> Index(ind.tag, max(0, pl)), cd(kw), plev)
noprime!(LT::LurTensor; kw...) = modify!(LT, (ind) -> Index(ind.tag, 0), cd(kw))
mapprime!(LT::LurTensor, plold::Int, plnew::Int; kw...) = modify!(LT, map_prime, cd(kw), plold, plnew)
swapprime!(LT::LurTensor, pl1::Int, pl2::Int; kw...) = modify!(LT, swap_prime, cd(kw), pl1, pl2)

# Modify plevs of LurTensor object(Copy version)
Base.adjoint(LT::LurTensor) = LurTensor(LT.arr, [i' for i in LT.inds])
prime(LT::LurTensor, plinc::Int = 1; kw...) = modify(prime!, LT, plinc; kw...)
setprime(LT::LurTensor, plev::Int = 1; kw...) = modify(setprime!, LT, plev; kw...)
noprime(LT::LurTensor; kw...) = modify(noprime!, LT; kw...)
mapprime(LT::LurTensor, plold::Int, plnew::Int; kw...) = modify(mapprime!, LT, plold, plnew; kw...)
swapprime(LT::LurTensor, pl1::Int, pl2::Int; kw...) = modify(swapprime!, LT, pl1, pl2; kw...)

modify(ft, LT::LurTensor, arg...; kw...) = (LT2 = copy(LT); ft(LT2, arg...; kw...))

function modify!(LT::LurTensor{T, N}, ft::Function, cond_ft, arg...) where {T, N} 
	for i=1:N
		if cond_ft(LT.inds[i])
			LT.inds[i] = ft(LT.inds[i], arg...)
		end
	end
	return LT
end


# Permutedim
permute_vec(v::Vector, vi) = [v[vi[i]] for i=1:length(vi)]

get_pvec(linds::Vector{Int}, ndim::Int) = vcat(linds, [i for i=1:ndim if !(i in linds)])
function Base.permutedims(LT::LurTensor{T, N}, perm) where {T, N}
	new_arr = permutedims(LT.arr, perm)
	new_inds = permute_vec(LT.inds, perm)
	return LurTensor(new_arr, new_inds)
end

# Contract
Base.sort(tag::String) = join(sort(split(tag, ',')), ',')
function get_dim(LT::LurTensor, i::Index) 
	d = findfirst(x -> x == i, LT.inds)
	if d isa Nothing
		error("There are no index $(i) in LurTensor")
	end
	return d
end
get_dim(LT::LurTensor, is::Vector{Index}) = [get_dim(LT, i) for i in is]


function get_contract_info(inds)
	elem_count = Dict{Index, Vector{Int}}()
	for (i, ind) in enumerate(inds)
		if ind in keys(elem_count)
			push!(elem_count[ind], i)
		else
			elem_count[ind] = [i]
		end
	end
	raxis, caxis1, caxis2 = Vector{Int}(), Vector{Int}(), Vector{Int}()
	
	for ind in keys(elem_count)
		if length(elem_count[ind]) > 2
			error("There are three (or more) Index objects with same tag and plev")
		elseif length(elem_count[ind]) == 2
			i, j = elem_count[ind]; push!(caxis1, i); push!(caxis2, j)
		else
			push!(raxis, elem_count[ind][1])
		end
	end
	return sort(raxis), caxis1, caxis2, length(caxis1)
end

Base.:+(x::LurTensor, y::LurTensor) = tensor_arith(x, y, Base.:+)
Base.:-(x::LurTensor, y::LurTensor) = tensor_arith(x, y, Base.:-)
Base.:*(x::LurTensor, y::LurTensor) = contract(contract(x), contract(y))

new_dim1(sz, raxis, caxis1, caxis2) = [new_dim1(sz, raxis, Val(length(raxis)))..., prod(sz[caxis1]), prod(sz[caxis2])]
new_dim1(sz, raxis, ::Val{0}) = []
new_dim1(sz, raxis, ::Any) = [prod(sz[raxis])]

new_dim2(sz, pvec, n, ::Any, ::Val{true}) = [prod(sz)]
new_dim2(sz, pvec, n, ::Val{'L'}, ::Val{false}) = [prod(sz[pvec[1:n]]), prod(sz[pvec[n+1:end]])]
new_dim2(sz, pvec, n, ::Val{'R'}, ::Val{false}) = [prod(sz[pvec[1:end-n]]), prod(sz[pvec[end-n+1:end]])]

mult(arr1::Array, arr2) = arr1 * arr2
mult(arr1::Vector, arr2) = transpose(transpose(arr1) * arr2)
mult(arr1::Vector, arr2::Vector) = transpose(arr1) * arr2

function contract(LT::LurTensor)
	raxis, caxis1, caxis2, nc = get_contract_info(LT.inds)
	sz = size(LT)
	if length(caxis1) == 0
		return LT
	end
	new_dim = new_dim1(sz, raxis, caxis1, caxis2)
	arr_permuted = permutedims(LT.arr, vcat(raxis, caxis1, caxis2))
	arr_reshaped = reshape(arr_permuted, new_dim...)
	if length(raxis) == 0
		return LurTensor(tr(arr_reshaped))
	end
	new_inds = permute_vec(LT.inds, raxis)

	output = zeros(new_dim[1])
	for i=1:new_dim[1]
		output[i] = tr(arr_reshaped[i, :, :])
	end
	output = reshape(output, sz[raxis]...)
	LurTensor(output, new_inds)
end

function contract(LT1::LurTensor, LT2::LurTensor)
	all_inds = vcat(LT1.inds, LT2.inds)
	raxis, caxis1, caxis2, nc = get_contract_info(all_inds)
	contract(LT1.arr, LT1.inds, LT2.arr, LT2.inds, raxis, caxis1, caxis2, nc)
end

function contract(arr1, inds1, arr2, inds2, raxis, caxis1, caxis2, nc)
	sz1, sz2 = size(arr1), size(arr2); sz = vcat(sz1..., sz2...)
	d1, d2 = length(sz1), length(sz2)
	permute_info = vcat(raxis[1:d1-nc], caxis1, caxis2, raxis[d1-nc+1:end])
	arr1_permuted = permutedims(arr1, permute_info[1:d1])
	arr2_permuted = permutedims(arr2, permute_info[d1+1:end].-d1)

	new_dim_l = new_dim2(sz1, permute_info[1:d1], d1-nc, Val('L'), Val(d1==nc))
	new_dim_r = new_dim2(sz2, permute_info[d1+1:end].-d1, d2-nc, Val('R'), Val(d2==nc))
	arr1_reshaped = reshape(arr1_permuted, new_dim_l...)
	arr2_reshaped = reshape(arr2_permuted, new_dim_r...)

	new_inds = permute_vec(vcat(inds1, inds2), raxis)
	new_sz = permute_vec(sz, raxis)
	m = mult(arr1_reshaped, arr2_reshaped)
	if m isa Number
		return LurTensor(m)
	end
	output = reshape(m, new_sz...)
	LurTensor(output, new_inds)
end

function tensor_arith(x, y, ft)
	if length(noncommoninds(x, y)) > 0
		error("Two tensors don't have same set of indices")
	end
	permute_vec = Vector{Int}()
	for ind in y.inds
		xi = findfirst(x -> x == ind, x.inds)
		if xi in permute_vec
			error("There are same indices within a tensor")
		end
		push!(permute_vec, xi)
	end
	LurTensor(ft(x.arr, permutedims(y.arr, permute_vec)), x.inds)
end


commoninds(LT1, LT2; kw...) = inds_meet_cond(intersect, LT1, LT2; kw...)
uniqueinds(LT1, LT2; kw...) = inds_meet_cond(setdiff, LT1, LT2; kw...)
noncommoninds(LT1, LT2; kw...) = inds_meet_cond(symdiff, LT1, LT2; kw...)
unioninds(LT1, LT2; kw...) = inds_meet_cond(union, LT1, LT2; kw...)
inds(LT1::LurTensor; kw...) = inds_meet_cond(x -> x, LT1; kw...)
hascommoninds(LT1, LT2; kw...) = isempty(commoninds(LT1, LT2; kw))
hascommoninds(LT1; kw...) = (LT2) -> hascommoninds(LT1, LT2; kw...)

commonind(LT1, LT2; kw...) = first_elem(commoninds(LT1, LT2; kw...))
uniqueind(LT1, LT2; kw...) = first_elem(uniqueinds(LT1, LT2; kw...))
noncommonind(LT1, LT2; kw...) = first_elem(noncommoninds(LT1, LT2; kw...))
unionind(LT1, LT2; kw...) = first_elem(unioninds(LT1, LT2; kw...))
ind(LT1::LurTensor; kw...) = first_elem(inds(LT1; kw...))

# tested only for reaplceinds! function
replaceinds!(LT::LurTensor, i1, i2) = replinds!(LT, get_map(i1, i2; bidir=false))
swapinds!(LT::LurTensor, i1, i2) = replinds!(LT, get_map(i1, i2; bidir=true))

replaceinds(LT::LurTensor, i1, i2) = replaceinds!(shallow_copy(LT), i1, i2)
swapinds(LT::LurTensor, i1, i2) = swapinds!(shallow_copy(LT), i1, i2)


replaceind!(LT::LurTensor, i1, i2) = replaceinds!(LT, [i1], [i2])
swapind!(LT::LurTensor, i1, i2) = swapinds!(LT, [i1], [i2])

replaceind(LT::LurTensor, i1, i2) = replaceind!(shallow_copy(LT), i1, i2)
swapind(LT::LurTensor, i1, i2) = swapind!(shallow_copy(LT), i1, i2)

hconj(LT::LurTensor{T, 2}) where {T} = conj(swapind(LT, LT.inds...))

get_map(i1, i2; kw...) = get_map([Index(i) for i in i1], [Index(i) for i in i2]; kw...)

function get_map(i1::Vector{Index}, i2::Vector{Index}; bidir=false)
	@assert length(i1) == length(i2)
	map = Dict{Index, Index}()
	for i=1:length(i1)
		map[i1[i]] = i2[i]
		if bidir && !(i2[i] in keys(map))
			map[i2[i]] = i1[i]
		end
	end
	return map
end

function replinds!(LT::LurTensor, replace_map::Dict{Index, Index})
	for (i, ind) in enumerate(LT.inds)
		if ind in keys(replace_map)
			LT.inds[i] = replace_map[ind]
		end
	end
	return LT
end

idx_list(A::LurTensor) = A.inds
idx_list(v::Vector{Index}) = v

inds_meet_cond(ft, LTs...; kw...) = 
	inds_meet_cond(ft, map(x -> idx_list(x), LTs)...; kw...)

inds_meet_cond(ft, Idxvs::Vector{Index}...; kw...) = 
	filter(x -> cd(kw)(x), ft(Idxvs...))

first_elem(v::Vector) = isempty(v) ? nothing : v[1]

decomp(LT, ft, linds::Vector{Index}; kw...) = decomp(LT, ft, get_dim(LT, linds); kw...)
decomp(LT, ft, linds::Int...; kw...) = decomp(LT, ft, collect(linds); kw...)
decomp(LT, ft, linds...; kw...) = decomp(LT, ft, to_indvec(linds...); kw...)

LinearAlgebra.eigen(LT::LurTensor, linds...; kw...) = decomp(LT, eigen_, linds...; kw...)
LinearAlgebra.qr(LT::LurTensor, linds...; kw...) = decomp(LT, qr_, linds...; kw...)
LinearAlgebra.svd(LT::LurTensor, linds...; kw...) = decomp(LT, svd_, linds...; kw...)
LinearAlgebra.eigvals(LT::LurTensor, linds...; kw...) = decomp(LT, eigvals_, linds...; kw...)
LinearAlgebra.diag(LT::LurTensor) = LinearAlgebra.diag(LT.arr)

# TODO: make methods for some classes of matrices
eigen_(mat::Matrix; kw...) = (e = LinearAlgebra.eigen(mat); [e.vectors, diagm(e.values), LinearAlgebra.inv(e.vectors)])
eigen_(mat::Hermitian; kw...) = (e = LinearAlgebra.eigen(mat); [e.vectors, diagm(e.values), adjoint(e.vectors)])
qr_(mat::Matrix; kw...) = ((q, r) = LinearAlgebra.qr(mat); [Matrix(q), r])
function svd_(mat::Matrix; kw...)
	u, s, v = LinearAlgebra.svd(mat)
	cutoff = get(kw, :cutoff, 0)
	# truncation index
	ti = findfirst(x -> x <= cutoff, s)
	ti = ti == nothing ? length(s) : ti - 1
	[u[:,1:ti], diagm(s[1:ti]), adjoint(v)[1:ti,:]]
end
eigvals_(mat::Matrix) = LinearAlgebra.eigvals(mat)

# TODO: Add code that exploit the symmetry of target LurTensor
# Symmetry is given by kw
function decomp(LT, ft, linds::Vector{Int}; kw...)
	# Reshape the array to matrix form
	nl = length(linds); pvec = get_pvec(linds, ndims(LT))
	LT_t = permutedims(LT, pvec); 
	sz = size(LT_t); ls, rs = prod(sz[1:nl]), prod(sz[nl+1:end])
	to_be_decomp = reshape(LT_t.arr, ls, rs)
	# decompose (ft can be qr, svd, eigen...)
	decomp_res = ft(to_be_decomp; kw...)
	if eltype(decomp_res) <: Number
		return decomp_res
	end
	
	# reshape resultant matrices to LurTensor objects
	lmarr, marr..., rmarr = decomp_res
	addtag = get(kw, :addtag, ""); plev = get(kw, :plev, 0)
	lmarr_reshaped = reshape(Matrix(lmarr), (sz[1:nl]..., size(lmarr)[end]))
	rmarr_reshaped = reshape(Matrix(rmarr), (size(rmarr)[1], sz[nl+1:end]...))
	mid_tags = get_mtags(String(Symbol(ft))[1:end-1], addtag, length(marr)) 
	lmLT = LurTensor(lmarr_reshaped, [LT_t.inds[1:nl]..., Index(mid_tags[1], plev)])
	rmLT = LurTensor(rmarr_reshaped, [Index(mid_tags[end], plev), LT_t.inds[nl+1:end]...])
	mLT = [LurTensor(Matrix(arr), [mid_tags[i], mid_tags[i+1]], [plev, plev]) for (i, arr) in enumerate(marr)]
	return [lmLT, mLT..., rmLT]
end


function get_mtags(ftname, addtag::String, n::Int)
	add_tag = length(addtag) != 0
	tags = (n == 0) ? [",bond"] : [",left", ",right"]
	return [add_tag ? ftname*i*","*addtag : ftname*i for i in tags]
end


include("GetIdentity.jl")
include("LocalSpace.jl")
include("UpdateLeft.jl")
