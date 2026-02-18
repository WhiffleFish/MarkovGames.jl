struct MGSimIterator{SPEC, M<:MG, P<:Policy, RNG<:AbstractRNG, S}
    game::M
    policy::P
    rng::RNG
    init_state::S
    max_steps::Int
end

function MGSimIterator(spec::Union{Tuple, Symbol}, game::MG, policy::Policy, rng::AbstractRNG, init_state, max_steps::Int) 
    return MGSimIterator{spec, typeof(game), typeof(policy), typeof(rng), typeof(init_state)}(game, policy, rng, init_state, max_steps)
end

Base.IteratorSize(::Type{<:MGSimIterator}) = Base.SizeUnknown()

function Base.iterate(it::MGSimIterator, is::Tuple{Int, S}=(1, it.init_state)) where S
    if isterminal(it.game, is[2]) || is[1] > it.max_steps 
        return nothing 
    end 
    t = is[1]
    s = is[2]
    σ, σ_info = behavior_info(it.policy, s)
    a = rand(it.rng, σ)
    out = @gen(:sp,:r,:info)(it.game, s, a, it.rng)
    nt = merge(NamedTuple{(:sp,:r,:info)}(out), (t=t, s=s, a=a, behavior=σ, behavior_info=σ_info))
    return (Simulators.out_tuple(it, nt), (t+1, nt.sp))
end

function Simulators.out_tuple(::MGSimIterator{spec}, all::NamedTuple) where spec
    if isa(spec, Tuple)
        return NamedTupleTools.select(all, spec)
    else 
        @assert isa(spec, Symbol) "Invalid specification: $spec is not a Symbol or Tuple."
        return all[spec]
    end
end

Simulators.default_spec(game::MG) = Simulators.default_spec(typeof(game))
Simulators.default_spec(::Type{<:MG}) = tuple(:s, :a, :sp, :r, :info, :behavior, :behavior_info, :t)
Simulators.convert_spec(spec, ::Type{<:MG}) = convert_spec(spec, Set(tuple(:s, :a, :sp, :r, :info, :behavior, :behavior_info, :t)))
Simulators.convert_spec(::Simulators.CompleteSpec, T::Type{<:MG}) = default_spec(T)

function POMDPs.value(hist::SimHistory)
    γ = hist.discount
    mapreduce(+, hist, eachindex(hist)) do h_i, t
        return h_i.r * γ ^ (t-1)
    end
end
