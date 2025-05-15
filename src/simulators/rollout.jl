function simulate(sim::RolloutSimulator, mdp::MDP, policy::Policy)
    istate = rand(sim.rng, initialstate(mdp))
    simulate(sim, mdp, policy, istate)
end

# FIXME: hacky and bad
function reward_type(::G) where {G<:Game}
    return first(Base.return_types(@gen(:r), Tuple{G, statetype(G), actiontype(G)}))
end

Base.zero(::Type{NTuple{N, T}}) where {N,T<:Number} = ntuple(i->zero(T), Val{N}())


#= 
FIXME: we should have some standardized reward return type.
Currently, sometimes I assume a tuple. Other times, in example games I just return
a single float because it's assumed to be zero-sum. This is confusing and annoying.
=#
function POMDPs.simulate(sim::RolloutSimulator, game::MG{S}, policy::Policy, s::S; rt=reward_type(game)) where {S}
    (;rng, max_steps) = sim

    γt = 1.0
    γ = discount(game)
    r_total = zero(rt)
    step = 1

    while γt > sim.eps && !isterminal(game, s) && step ≤ max_steps
        a = rand(rng, behavior(policy, s))
        sp, r = @gen(:sp,:r)(game, s, a, rng)
        r_total = r_total .+ γt .* r

        s = sp
        γt *= γ
        step += 1
    end

    return r_total
end
