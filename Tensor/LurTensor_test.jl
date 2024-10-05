include("LurTensor.jl")

"""
lur = [1 2; 3 4]
asdf = LurTensor(lur, ["as", "aa"], [0, 0])
println(asdf)
for i=1:4
	asdf[i] = i + 4
end
println(asdf)


aaa = LurTensor(rand(3, 4, 5), ["AA", "BB", "CC"])
aaap = aaa[1:2, 2:3, 2:3]

show(stdout, aaa)
show(stdout, aaap)

show(stdout, aaa .+ 2)
println(typeof(aaa .+ 2))

println("\n\nTest modify function\n\n")

aaaa = LurTensor(rand(2, 3, 2, 4), ["A,AA,AAA", "Q,WTW,AAA", "B,BB,AA", "WTW,CC,C"], [0, 1, 2, 3])
show(stdout, aaaa)

println("1. change tags (not in-place)")
aaaar = removetags(aaaa, "AA")
show(stdout, removetags(aaaa, "AA"))

aaaar[2:5] = 4:7
show(stdout, aaaa; showarr=true)  # Should not be changed
show(stdout, aaaar; showarr=true) # Should be changed

println("2. change tags (in-place)")
aaarr = removetags!(aaaa, "AAA,WTW")
aaarr[2:5] = 4:7
show(stdout, aaaa, showarr=true) # Should be changed


println("3. Do not allow empty tags")
try
	removetags!(aaaa, "Q")
	println("This should not be printed")
catch
	println("This should be printed")
end

println("4. Add / Swap tags")
aaaa = LurTensor(rand(2, 3, 2, 4), ["A,AA,AAA", "Q,WTW,AAA", "B,BB,AA", "WTW,CC,C"], [0, 1, 2, 3])
println("Before add")
show(stdout, aaaa)
println("\n\nAfter add")
addtags!(aaaa, "AAA,AA")
show(stdout, aaaa)

println("5. test condition")
aaaa = LurTensor(rand(2, 3, 2, 4), ["A,AA,AAA", "Q,WTW,AAA", "B,BB,AA", "WTW,CC,C"], [0, 1, 0, 1])
println("Before add")
show(stdout, aaaa)

println("\n\nAfter add")
addtags!(aaaa, "AAA,AA,QERQ"; plev=1, tag="Q") # added only to dim 2
show(stdout, aaaa)
addtags!(aaaa, "B,E,F"; tag="AAA,AA") # added to dim 1, dim 2
show(stdout, aaaa)
replacetags!(aaaa, "A,E", "LURLUR") # replaced in dim 1
show(stdout, aaaa)
replacetags!(aaaa, "AA", "DAE"; plev=1) # replaced in dim 2
show(stdout, aaaa)
replacetags!(aaaa, "B", "BBBB"; plev=1) # replaced in dim 2
show(stdout, aaaa)
settags!(aaaa, "ASD,LGSL"; plev=1) # set in dim 2, 4
show(stdout, aaaa)
swaptags!(aaaa, "AA,B", "LGSL") # affect all dims
show(stdout, aaaa)
replacetags!(aaaa, "AAA", "ASD") # affect dim 1
show(stdout, aaaa)
removetags!(aaaa, "ASD"; tag="LURLUR")
show(stdout, aaaa)
prime!(aaaa, 3; tag="F")
show(stdout, aaaa)
setprime!(aaaa, 2; plev=1)
show(stdout, aaaa)
noprime!(aaaa; tag="LURLUR")
show(stdout, aaaa)
mapprime!(aaaa, 0, 3; tag="F")
show(stdout, aaaa)
mapprime!(aaaa, 2, 3)
show(stdout, aaaa)
swapprime!(aaaa, 0, 3)
show(stdout, aaaa)
swapprime!(aaaa, 0, 1)
show(stdout, aaaa)
swapprime!(aaaa, 2, 3)
show(stdout, aaaa)
mapprime!(aaaa, 2, 1)
show(stdout, aaaa)

try
	a = LurTensor(rand(2, 3), ["", "a"])
	println("This should not be printed")
catch
	println("This should be printed")
end

try
	settags!(aaaa, ""; plev=0)
	println("This should not be printed")
catch
	println("This should be printed")
end

# Permute dimension
p = permutedims(aaaa, [2, 4, 3, 1])
show(stdout, p)

a = reshape(1:48, 2, 3, 2, 4)
lt = LurTensor(a, ["a", "b", "a", "c"])
show(stdout, lt, showarr=true)
lt = contract(lt)
show(stdout, lt, showarr=true)

a = rand(2, 1, 1, 2, 2, 2)
lt = LurTensor(a, ["a", "C", "D", "b", "b", "a"])
show(stdout, lt, showarr=true)
lt = contract(lt)
show(stdout, lt, showarr=true)
"""

a = reshape(1:24, 3, 2, 4)
l = LurTensor(a, ["B", "A", "C"])
b = reshape(1:8, 4, 2)
r = LurTensor(b, ["C", "A"])
c = l * r
show(stdout, l, showarr=true)
show(stdout, r, showarr=true)
show(stdout, c, showarr=true)


a = reshape(1:6, 2, 3)
l = LurTensor(a, ["B", "C"])
b = reshape(1:24, 3, 4, 2)
r = LurTensor(b, ["C", "A", "B"])
c = l * r
show(stdout, l, showarr=true)
show(stdout, r, showarr=true)
show(stdout, c, showarr=true)

a = reshape(1:24, 4, 2, 3, 1)
l = LurTensor(a, ["F", "B", "C", "D"])
b = reshape(1:30, 3, 5, 1, 2)
r = LurTensor(b, ["C", "A", "G", "B"])
c = l * r
show(stdout, l, showarr=true)
show(stdout, r, showarr=true)
show(stdout, c, showarr=true)

a = reshape(1:24, 4, 2, 3)
l = LurTensor(a, ["F,B", "B,A", "C,DWD"])
b = reshape(1:24, 2, 3, 4)
r = LurTensor(b, ["A,B", "DWD,C", "C,F"])
c = l * r
show(stdout, l, showarr=true)
show(stdout, r, showarr=true)
show(stdout, c, showarr=true)

a = reshape(1:6, 2, 3)
l = LurTensor(a, ["LUR", "LALA"])
b = reshape(1:6, 2, 3)
r = LurTensor(b, ["LALLLA", "DWD,C"])
c = l * r
show(stdout, l, showarr=true)
show(stdout, r, showarr=true)
show(stdout, c, showarr=true)

