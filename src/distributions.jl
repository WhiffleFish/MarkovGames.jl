struct ProductDistribution{D<:Union{Tuple, AbstractArray}}
    dists::D
    ProductDistribution(t::Union{Tuple,AbstractArray}) = new{typeof(t)}(t)
    ProductDistribution(args...) = new{typeof(args)}(args)
end

Base.iterate(d::ProductDistribution, args...) = iterate(d.dists, args...)
Base.getindex(d::ProductDistribution, i) = getindex(d.dists, i)
Base.length(d::ProductDistribution) = length(d.dists)

POMDPs.pdf(d::ProductDistribution, x) = mapreduce(*, x, d.dists) do x_i, dist
    POMDPs.pdf(dist, x_i)
end

# loop unrolling
POMDPs.pdf(d::ProductDistribution{<:NTuple{2,Any}}, x) = POMDPs.pdf(d[1], x[1]) * POMDPs.pdf(d[2], x[2])

POMDPs.support(d::ProductDistribution) = Iterators.product((support(_d) for _d ∈ d)...)

Random.rand(rng::AbstractRNG, d::Random.SamplerTrivial{<:ProductDistribution}) = map(_d->rand(rng, _d), d[].dists)
