"""
    DiscretePOMGBelief

A belief specified by a probability vector.

Normalization of `b` is assumed in some calculations (e.g. pdf), but it is only automatically enforced in `update(...)`, and a warning is given if normalized incorrectly in `DiscreteBelief(pomdp, b)`.

# Constructor
    DiscretePOMGBelief(game::POMG, b::Vector{Float64}; check::Bool=true)

# Fields 
- `game` : the POMG problem  
- `state_list` : a vector of ordered states
- `b` : the probability vector 
"""
struct DiscretePOMGBelief{P<:POMG, S}
    game::P
    state_list::Vector{S}       # vector of ordered states
    b::Vector{Float64}
end

function POMGs.isterminal_belief(game::POMG, b::DiscretePOMGBelief)
    for (s,p) ∈ zip(b.state_list, b.b)
        isterminal(game, s) && !iszero(p) && return true
    end
    return false
end

BeliefUpdaters.DiscreteBelief(game::POMG, b::Vector{Float64}; check::Bool=true) = DiscretePOMGBelief(game, b; check)

function DiscretePOMGBelief(game::POMG, b::Vector{Float64}; check::Bool=true)
    if check
        if !isapprox(sum(b), 1.0, atol=0.001)
            @warn("""
                  b in DiscreteBelief(pomdp, b) does not sum to 1.
 
                  To suppress this warning use `DiscreteBelief(pomdp, b, check=false)`
                  """, b)
        end
        if !all(0.0 <= p <= 1.0 for p in b)
            @warn("""
                  b in DiscreteBelief(pomdp, b) contains entries outside [0,1].
 
                  To suppress this warning use `DiscreteBelief(pomdp, b, check=false)`
                  """, b)
        end
    end
    return DiscretePOMGBelief(game, ordered_states(game), b)
end


"""
     uniform_belief(pomdp)

Return a DiscreteBelief with equal probability for each state.
"""
function BeliefUpdaters.uniform_belief(game::POMG)
    state_list = ordered_states(game)
    ns = length(state_list)
    return DiscreteBelief(game, state_list, ones(ns) ./ ns)
end

Distributions.pdf(b::DiscretePOMGBelief, s) = b.b[stateindex(b.pomdp, s)]

function Random.rand(rng::Random.AbstractRNG, b::DiscretePOMGBelief)
    i = sample(rng, Weights(b.b))
    return b.state_list[i]
end

Base.fill!(b::DiscretePOMGBelief, x::Float64) = fill!(b.b, x)

Base.length(b::DiscretePOMGBelief) = length(b.b)

Distributions.support(b::DiscretePOMGBelief) = b.state_list

Statistics.mean(b::DiscretePOMGBelief) = sum(b.state_list .* b.b)/sum(b.b)
StatsBase.mode(b::DiscretePOMGBelief) = b.state_list[argmax(b.b)]

Base.:(==)(b1::DiscretePOMGBelief, b2::DiscretePOMGBelief) = b1.state_list == b2.state_list && b1.b == b2.b

Base.hash(b::DiscretePOMGBelief, h::UInt) = hash(b.b, hash(b.state_list, h))

"""
    DiscreteUpdater

An updater type to update discrete belief using the discrete Bayesian filter.

# Constructor
    DiscreteUpdater(game::POMG)

# Fields
- `game <: POMG`
"""
struct DiscretePOMGUpdater{P<:POMG} <: POMDPs.Updater
    game::P
end

BeliefUpdaters.DiscreteUpdater(game::POMG) = DiscretePOMGUpdater(game)

BeliefUpdaters.uniform_belief(up::DiscretePOMGUpdater) = uniform_belief(up.game)

function POMDPs.initialize_belief(bu::DiscretePOMGUpdater, dist::Any)
    state_list = ordered_states(bu.game)
    ns = length(state_list)
    b = zeros(ns)
    belief = DiscreteBelief(bu.game, state_list, b)
    for s in support(dist)
        sidx = stateindex(bu.game, s)
        belief.b[sidx] = pdf(dist, s)
    end
    return belief
end

function POMDPs.update(bu::DiscretePOMGUpdater, b::DiscretePOMGBelief, a, o)
    game = bu.game
    state_space = b.state_list
    bp = zeros(length(state_space))

    for (si, s) in enumerate(state_space)

        if pdf(b, s) > 0.0
            td = transition(game, s, a)

            for (sp, tp) in weighted_iterator(td)
                spi = stateindex(game, sp)
                op = obs_weight(game, s, a, sp, o) # shortcut for observation probability from POMDPModelTools

                bp[spi] += op * tp * b.b[si]
            end
        end
    end

    bp_sum = sum(bp)

    if bp_sum == 0.0
        error("""
              Failed discrete belief update: new probabilities sum to zero.

              b = $b
              a = $a
              o = $o
              """)
    end

    # Normalize
    bp ./= bp_sum

    return DiscretePOMGBelief(pomdp, b.state_list, bp)
end

POMDPs.update(bu::DiscretePOMGUpdater, b::Any, a, o) = update(bu, initialize_belief(bu, b), a, o)
