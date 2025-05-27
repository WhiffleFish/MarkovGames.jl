export 
    Policy,
    behavior,
    behavior_info,
    solve

"""
    solve(solver, game::Game)

Solves the Game using method associated with solver, and returns a policy.
"""
function POMDPs.solve(solver, game::Game) end

"""
    behavior(pol::Policy, s)

Yields distribution over actions from the associated policy for a given state
"""
function behavior end

function behavior_info end

behavior_info(pol, s) = (behavior(pol, s), (;))
