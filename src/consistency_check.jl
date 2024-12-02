function POMDPTools.has_consistent_distributions(m::POMG; atol=0.0)
    return POMDPTools.has_consistent_initial_distribution(m; atol) &&
        POMDPTools.has_consistent_transition_distributions(m; atol) &&
        POMDPTools.has_consistent_observation_distributions(m; atol)
end

function POMDPTools.has_consistent_distributions(m::MG; atol=0.0)
    return POMDPTools.has_consistent_initial_distribution(m; atol) &&
        POMDPTools.has_consistent_transition_distributions(m; atol)
end

function POMDPTools.has_consistent_transition_distributions(game::Game; atol=0.0)
    S = states(game)
    A1, A2 = actions(game)
    ok = true
    for s ∈ S
        for a1 ∈ A1, a2 ∈ A2
            a = (a1, a2)
            d = transition(game, s, a)
            psum = 0.0
            sup = Set(support(d))
            for sp in sup
                if POMDPs.pdf(d, sp) > 0.0 && !(sp ∈ S)
                    @warn "sp in support(transition(m, s, a)), but not in states(game)" s a sp
                    ok = false
                end
            end
            for sp in S 
                p = POMDPs.pdf(d, sp)
                if p < 0.0
                    @warn "Transition probability negative ($p < 0.0)." s a sp
                    ok = false
                elseif p > 0.0 && !(sp in sup)
                    @warn "State $sp with probability $p is not in support" s a
                    ok = false
                end
                psum += p
            end
            if !isapprox(psum, 1.0; atol)
                @warn "Transition probabilities sum to $psum, not 1. Consider atol keyword argument." s a
                ok = false
            end
        end
    end
    return ok
end

function POMDPTools.has_consistent_observation_distributions(game::POMG; atol=0.0)
    S = states(game)
    O1, O2 = observations(game)
    A1, A2 = actions(game)
    ok = true
    for s in states(game)
        if !isterminal(game, s)
            for a1 ∈ A1, a2 ∈ A2
                a = (a1, a2)
                for sp in S
                    obs = observation(game, s, a, sp)
                    psum = 0.0
                    sup = Set(support(obs))
                    for o in sup
                        o1, o2 = o
                        if POMDPs.pdf(obs, o) > 0.0 && !(o1 in O1 && o2 ∈ O2)
                            @warn "o in support(observation(game, s, a, sp)), but not in observations(game)" s a sp o
                            ok = false
                        end
                    end
                    for o1 ∈ O1, o2 ∈ O2
                        o = (o1,o2)
                        p = POMDPs.pdf(obs, o)
                        if p < 0.0
                            @warn "Observation probability negative ($p < 0.0)." s a sp o
                            ok = false
                        elseif p > 0.0 && !(o in sup)
                            @warn "Observation $o with probability $p is not in support." s a sp
                            ok = false
                        end
                        psum += p
                    end
                    if !isapprox(psum, 1.0; atol)
                        @warn "Observation probabilities sum to $psum, not 1. Consider atol keyword argument." s a sp
                        ok = false
                    end
                end
            end
        end
    end
    return ok
end

function POMDPTools.has_consistent_initial_distribution(game::Game; atol=0.0)
    ok = true
    d = initialstate(game)
    sup = Set(support(d))
    psum = 0.0
    for s in states(game)
        p = POMDPs.pdf(d, s)
        psum += p
        if p < 0.0
            @warn "Initial state probability negative ($p < 0.0)" s
            ok = false
        elseif p > 0.0 && !(s in sup)
            @warn "State $s with probability $p is not in initial distribution support."
            ok = false
        end
    end
    if !isapprox(psum, 1.0; atol)
        @warn "Initial state probabilities sum to $psum, not 1. Consider atol keyword argument."
        ok = false
    end
    return ok
end

