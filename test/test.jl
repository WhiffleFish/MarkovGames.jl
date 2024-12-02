using POMGs
using POMGs.Games
using POMDPs
using POMDPTools
using BenchmarkTools

dist_vec2 = [BoolDistribution(0.2), BoolDistribution(0.4)]
dist_tup2 = (BoolDistribution(0.2), BoolDistribution(0.4))
dist_tup2_2 = (BoolDistribution(0.2), SparseCat(['a', 'b'], [0.3, 0.7]))

d1 = ProductDistribution(dist_vec2)
d2 = ProductDistribution(dist_tup2)
d2_2 = ProductDistribution(dist_tup2_2)
@btime pdf(d1, (false, false))
@btime pdf(d2, (false, false))
@btime pdf(d2_2, (false, 'a'))

dist_vec3 = [BoolDistribution(0.2), BoolDistribution(0.4), BoolDistribution(0.6)]
dist_tup3 = (BoolDistribution(0.2), BoolDistribution(0.4), BoolDistribution(0.6))
dist_tup3_2 = (BoolDistribution(0.2), BoolDistribution(0.4), SparseCat(['a', 'b'], [0.3, 0.7]))
d3 = ProductDistribution(dist_vec3)
d4 = ProductDistribution(dist_tup3)
d4_2 = ProductDistribution(dist_tup3_2)

@btime pdf(d2, (false, false))
@btime pdf(d2, (false, false))

@btime pdf(d3, (false, false, false))
@btime pdf(d4, (false, false, false))
@btime pdf(d4_2, (false, false, 'a'))



game = CompetitiveTiger()
s = Games.TIGER_LEFT
a = (Games.LISTEN, Games.LISTEN)
sp = s

o_dist = observation(game, a, sp)
p = 0.0
for o ∈ support(o_dist)
    p += pdf(o_dist, o)
end
pdf(o_dist, (Games.TIGER_LEFT, Games.TIGER_LEFT))
pdf(o_dist, (Games.TIGER_LEFT, Games.TIGER_RIGHT))
pdf(o_dist, (Games.TIGER_RIGHT, Games.TIGER_LEFT))
pdf(o_dist, (Games.TIGER_RIGHT, Games.TIGER_RIGHT))
p

using SparseArrays
mat = sprandn(10, 10, 0.5)
mat[1,1:end]

using POMDPTools
using POMGs

@code_warntype rand(d1, 10)
[rand(d1) for _ in 1:3]
only(Base.return_types(rand, (typeof(d1),)))
eltype(d1)


@profview for _ in 1:1_000_000
    rand(d1, 10)
end

@edit Base.return_types(rand, (typeof(d1),))
Base._return_type
eltype(d1[1])
d1[1]


_f(args...) = "x"*string(args...)

