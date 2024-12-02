# TODO: Any way to check if there's a one-to-one correspondance between history and state?
struct SingletonUpdater{G}
    game::G
end

POMGs.isterminal_belief(game::POMG, b::Deterministic) = isterminal(game, b.val)

function POMDPs.update(upd::SingletonUpdater, b::Deterministic, a, o)
    game = upd.game
    s = b.val
    T = transition(game, s, a)
    T_support = support(T)
    if isone(length(T_support))
        return only(T_support)
    else
        for sp ∈ T_support
            !iszero(observation(game, s, a, sp)) && return Deterministic(sp)
        end
        error("""
            Failed belief update - no nonzero observation probabilities

            s = $s
            a = $a
            o = $o
        """)
    end
end

POMGs.belief_reward(game::POMG, b::Deterministic, a) = reward(game, b.val, a)
