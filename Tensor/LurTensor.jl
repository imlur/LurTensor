using LinearAlgebra

struct LurTensor{T<:Number, N} <: AbstractArray{T, N}
	arr::AbstractArray{T, N}
	tags::Vector{String}
	plevs::Vector{Int}

	function LurTensor(arr::AbstractArray{T, N}, tags::Vector{String}, plevs::Vector{Int}) where {T, N}
		if length(tags) != N 
			error("Dimension of array and length of tags not match")
		elseif 0 in length.(tags)
			error("Some of tags are empty string")
		elseif length(plevs) != N
			error("Dimension of array and length of plevs not match")
		end
		new{T, N}(arr, tags, plevs)
	end
end

LurTensor(x::Number) = LurTensor([x], ["Null"])

function showDim(io::IO, LT::LurTensor{T, N}) where {T, N}
	sz = size(LT)
	println("\t\tSize\tplev\ttag")
	for d=1:N	
		println("Dim $(d) : \t$(sz[d]) \t$(LT.plevs[d]) \t$(LT.tags[d])")
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
	new_tags = [LT.tags[i] for i=1:N if !(I[i] isa Integer)]
	new_plevs = [LT.plevs[i] for i=1:N if !(I[i] isa Integer)]
	LurTensor(new_arr, new_tags, new_plevs)
end

Base.similar(LT::LurTensor{T, N}) where {N, T} = LurTensor(zeros(T, size(LT)...), LT.tags, LT.plevs)

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
	LurTensor(similar(Array{ElType}, axes(bc)), A.tags, A.plevs)
end

find_aac(bc::Base.Broadcast.Broadcasted) = find_aac(bc.args)
find_aac(args::Tuple) = find_aac(find_aac(args[1]), Base.tail(args))
find_aac(x) = x
find_aac(::Tuple{}) = nothing
find_aac(a::LurTensor, rest) = a
find_aac(::Any, rest) = find_aac(rest)

# Copy LurTensor object
Base.copy(LT::LurTensor) = LurTensor(Base.copy(LT.arr), Base.copy(LT.tags), Base.copy(LT.plevs))

# Add condition to modify functions below
cond_kwargs = (tag=nothing, plev=nothing)

# Condition Function
function cd(kw)
	for k in keys(kw)
		if !(k in (:tag, :plev)) error("Keyword argument $(k) is not supported") end
	end
	t = get(kw, :tag, nothing); p = get(kw, :plev, nothing)
	(lt, lp) -> check(lt, t) && check(lp, p)
end

check(tag::String, str::String) = all(x -> x in split(tag, ','), split(str, ','))
check(plev::Int, i::Int) = plev == i
check(::Any, ::Nothing) = true

# Change tags
function remove_tag(tag::String, str; check_empty=true)
	tag_split = split(tag, ',')
	for s in split(str, ',') 
		filter!(e->e!=s, tag_split)
	end
	if check_empty && length(tag_split) == 0
		error("Empty tag is not allowed")
	end
	return join(tag_split, ',')
end

set_tag(tag::String, str) = set_tag(tag, str, Val(length(str)))
set_tag(tag::String, str, ::Val{0}) = error("Empty tag is not allowed")
set_tag(tag::String, str, ::Any) = str

function add_tag(tag::String, str::String)
	tag_split = split(tag, ',')
	for s in split(str, ',')
		if !(s in tag_split)
			tag *= ',' * s 
		end
	end
	return tag
end

replace_tag(tag::String, s1, s2) = replace_tag(tag, s1, s2, Val(check(tag, s1)))
replace_tag(tag::String, s1, s2, ::Val{true}) = add_tag(remove_tag(tag, s1; check_empty=false), s2)
replace_tag(tag::String, s1, s2, ::Val{false}) = tag

swap_tag(tag::String, s1, s2) = swap_tag(tag::String, s1, s2, Val(check(tag, s1)), Val(check(tag, s2)))
swap_tag(tag::String, s1, s2, ::Val{true}, ::Val{false}) = replace_tag(tag, s1, s2, Val(true))
swap_tag(tag::String, s1, s2, ::Val{false}, ::Val{true}) = replace_tag(tag, s2, s1, Val(true))
swap_tag(tag::String, s1, s2, ::Any, ::Any) = tag

map_prime(plev, plold, plnew) = (plev == plold) ? plnew : plev
swap_prime(plev, pl1, pl2) = (plev == pl1) ? pl2 : (plev == pl2) ? pl1 : plev


# Modify tags of LurTensor object (Inplace version)
addtags!(LT::LurTensor, str; kw...) = modify!(LT, add_tag, :tags, cd(kw), str)
removetags!(LT::LurTensor, str; kw...) = modify!(LT, remove_tag, :tags, cd(kw), str)
settags!(LT::LurTensor, str; kw...) = modify!(LT, set_tag, :tags, cd(kw), str)
replacetags!(LT::LurTensor, s1, s2; kw...) = modify!(LT, replace_tag, :tags, cd(kw), s1, s2)
swaptags!(LT::LurTensor, s1, s2; kw...) = modify!(LT, swap_tag, :tags, cd(kw), s1, s2)

# Modify tags of LurTensor object (Copy version)
removetags(LT::LurTensor, str; kw...) = modify(removetags!, LT, str; kw...)
addtags(LT::LurTensor, str; kw...) = modify(addtags!, LT, str; kw)
settags(LT::LurTensor, str; kw...) = modify(settags!, LT, str; kw...)
replacetags(LT::LurTensor, str1, str2; kw...) = modify(replacetags!, LT, str1, str2; kw...)
swaptags(LT::LurTensor, str1, str2; kw...) = modify(swaptags!, LT, str1, str2; kw...)

# Modify plevs of LurTensor object (Inplace version)
prime!(LT::LurTensor, plinc::Int = 1; kw...) = modify!(LT, (p, pl) -> max(0, p + pl), :plevs, cd(kw), plinc)
setprime!(LT::LurTensor, plev::Int; kw...) = modify!(LT, (_, pl) -> max(0, pl), :plevs, cd(kw), plev)
noprime!(LT::LurTensor; kw...) = modify!(LT, (_) -> 0, :plevs, cd(kw))
mapprime!(LT::LurTensor, plold::Int, plnew::Int; kw...) = modify!(LT, map_prime, :plevs, cd(kw), plold, plnew)
swapprime!(LT::LurTensor, pl1::Int, pl2::Int; kw...) = modify!(LT, swap_prime, :plevs, cd(kw), pl1, pl2)

# Modify plevs of LurTensor object(Copy version)
prime(LT::LurTensor, plinc::Int = 1; kw...) = modify(prime!, LT, plinc; kw...)
setprime(LT::LurTensor, plev::Int = 1; kw...) = modify(setprime!, LT, plev; kw...)
noprime(LT::LurTensor; kw...) = modify(noprime!, LT; kw...)
mapprime(LT::LurTensor, plold::Int, plnew::Int; kw...) = modify(mapprime!, LT, plold, plnew; kw...)
swapprime(LT::LurTensor, pl1::Int, pl2::Int; kw...) = modify(swapprime!, LT, pl1, pl2; kw...)

modify(ft, LT::LurTensor, arg...; kw...) = (LT2 = copy(LT); ft(LT2, arg...; kw...))

function modify!(LT::LurTensor{T, N}, ft::Function, fn, cond_ft, arg...) where {T, N} 
	for i=1:N
		tag, plev = LT.tags[i], LT.plevs[i]
		if cond_ft(tag, plev)
			getfield(LT, fn)[i] = ft(getfield(LT, fn)[i], arg...)
		end
	end
	return LT
end


# Permutedim
permute_vec(v::Vector, vi::Vector{Int}) = [v[vi[i]] for i=1:length(vi)]

function Base.permutedims(LT::LurTensor{T, N}, perm) where {T, N}
	new_arr = permutedims(LT.arr, perm)
	new_tags, new_plevs = permute_vec(LT.tags, perm), permute_vec(LT.plevs, perm)
	return LurTensor(new_arr, new_tags, new_plevs)
end

# Contract
Base.sort(tag::String) = join(sort(split(tag, ',')), ',')

function get_contract_info(inds)
	inds_tag_sorted = [(sort(t), p) for (t, p) in inds]
	elem_count = Dict{Tuple{String, Int}, Vector{Int}}()
	for (i, pair) in enumerate(inds_tag_sorted)
		if pair in keys(elem_count)
			push!(elem_count[pair], i)
		else
			elem_count[pair] = [i]
		end
	end
	raxis, caxis1, caxis2 = Vector{Int}(), Vector{Int}(), Vector{Int}()
	
	for pair in keys(elem_count)
		if length(elem_count[pair]) > 2
			error("There are three (or more) axes with same tag and plev")
		elseif length(elem_count[pair]) == 2
			i, j = elem_count[pair]; push!(caxis1, i); push!(caxis2, j)
		else
			push!(raxis, elem_count[pair][1])
		end
	end
	return sort(raxis), caxis1, caxis2, length(caxis1)
end

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
	axis_props = collect(zip(LT.tags, LT.plevs))	
	raxis, caxis1, caxis2, nc = get_contract_info(axis_props)
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
	new_prop = permute_vec(axis_props, raxis)

	output = zeros(new_dim[1])
	for i=1:new_dim[1]
		output[i] = tr(arr_reshaped[i, :, :])
	end
	output = reshape(output, sz[raxis]...)
	LurTensor(output, map(collect, collect(zip(new_prop...)))...)
end

function contract(LT1::LurTensor, LT2::LurTensor)
	LT1_axis = collect(zip(LT1.tags, LT1.plevs))
	LT2_axis = collect(zip(LT2.tags, LT2.plevs))
	axis_props = vcat(LT1_axis, LT2_axis)
	raxis, caxis1, caxis2, nc = get_contract_info(axis_props)
	contract(LT1.arr, LT1_axis, LT2.arr, LT2_axis, raxis, caxis1, caxis2, nc)
end

function contract(arr1, axis1, arr2, axis2, raxis, caxis1, caxis2, nc)
	sz1, sz2 = size(arr1), size(arr2); sz = vcat(sz1..., sz2...)
	d1, d2 = length(sz1), length(sz2)
	permute_info = vcat(raxis[1:d1-nc], caxis1, caxis2, raxis[d1-nc+1:end])
	arr1_permuted = permutedims(arr1, permute_info[1:d1])
	arr2_permuted = permutedims(arr2, permute_info[d1+1:end].-d1)

	new_dim_l = new_dim2(sz1, permute_info[1:d1], d1-nc, Val('L'), Val(d1==nc))
	new_dim_r = new_dim2(sz2, permute_info[d1+1:end].-d1, d2-nc, Val('R'), Val(d2==nc))
	arr1_reshaped = reshape(arr1_permuted, new_dim_l...)
	arr2_reshaped = reshape(arr2_permuted, new_dim_r...)

	new_prop = permute_vec([axis1..., axis2...], raxis)
	new_sz = permute_vec(sz, raxis)
	m = mult(arr1_reshaped, arr2_reshaped)
	if m isa Number
		return LurTensor(m)
	end
	output = reshape(m, new_sz...)
	LurTensor(output, map(collect, collect(zip(new_prop...)))...)
end
