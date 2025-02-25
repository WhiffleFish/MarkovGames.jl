export 
    POMG,
    MG,
    Game,
    discount, 
    transition, 
    observation,
    player_observation,
    reward, 
    isterminal, 
    initialstate, 
    players,
    states,
    actions,
    player_actions, 
    observations,
    player_observations,
    @gen,
    gen,
    belief_reward,
    isterminal_belief,
    statetype,
    actiontype,
    obstype,
    solve
    

"""
    POMG{S,A,O}
Abstract base type for partially observable Markov Games.
    S: state type
    A: joint action type
    O: joint observation type
"""
abstract type POMG{S,A,O} end

"""
    MG{S,A}
Abstract base type for fully observable Markov Games.
    S: state type
    A: joint action type
"""
abstract type MG{S,A} end

const Game = Union{POMG, MG}

"""
    discount(m::Game)
Return the discount factor for the problem.
"""
function POMDPs.discount(::Game) end

"""
    transition(m::Game, state, action)
Return the transition distribution from the current state-joint-action pair.
If it is difficult to define the probability density or mass function explicitly, consider using `POMDPModelTools.ImplicitDistribution` to define a generative model.
"""
function POMDPs.transition(::Game, s, a) end

POMDPs.observation(game::POMG, a, sp) = observation(game, sp)

POMDPs.observation(game::POMG, s, a, sp) = observation(game, a, sp)

"""
    player_observation(m::POMG, i::Int, a, sp)

Return observation distribution for player `i`

"""
function player_observation end

player_observation(m::POMG, i::Int, s, a, sp) = player_observation(m, i, a, sp)
player_observation(m::POMG, i::Int, a, sp) = player_observation(m, i, sp)


function POMDPs.reward(::Game, s, a) end

POMDPs.reward(p::Game, s, a, sp, o) = reward(p, s, a, sp)
POMDPs.reward(p::Game, s, a, sp) = reward(p, s, a)
POMDPs.reward(p, s, a) = reward(p, s)

"""
    isterminal(m::Game, s)
Check if state `s` is terminal.
If a state is terminal, no actions will be taken in it and no additional rewards will be accumulated. Thus, the value function at such a state is, by definition, zero.
"""
POMDPs.isterminal(problem::Game, state) = false

"""
    initialstate(m::Game)
Return a distribution of initial states for POMG `m`.
If it is difficult to define the probability density or mass function explicitly, consider using `POMDPModelTools.ImplicitDistribution` to define a model for sampling.
"""
function POMDPs.initialstate(::Game) end

function players end

players(::POMG) = 1:2

"""
    actions(game)

Returns the action space for each player (A1, A2)

--
    actions(game, s)

Returns the actions that can be taken for each player (A1, A2) at state `s`

"""
function POMDPs.actions(::Game) end

POMDPs.actions(g::Game, s) = actions(g)

"""
    player_actions(game, p)

Returns the action space for player `p`

--
    player_actions(game, p, s)

Returns the actions that can be taken by player `p` at state `s`
"""
function player_actions end

player_actions(game::Game, i) = actions(game)[i]

player_actions(game::Game, i, s) = actions(game, s)[i]

"""
    states(game::Game)

Returns the state space of a given game
"""
function POMDPs.states(::Game) end

"""
    (O1, O2) = observations(game)

Returns the observation space of a given game for both players
"""
function POMDPs.observations(::POMG) end

POMDPs.observations(p::POMG, s) = observations(p)

"""
    Oi = player_observations(game::POMG, i::Int)

Returns the observation space of a given game for player `i`
"""
function player_observations end

player_observations(p::POMG, i::Int, s) = player_observations(p, i)

@inline other_player(i) = 3-i

"""
    belief_reward(game::POMG, b, a)
"""
function belief_reward end

"""
    isterminal_belief(game::POMG, b)
"""
function isterminal_belief end

POMDPs.statetype(::POMG{S}) where S = S
POMDPs.actiontype(::POMG{S,A}) where {S,A} = A
POMDPs.obstype(::POMG{S,A,O}) where {S,A,O} = O

POMDPs.statetype(::MG{S}) where S = S
POMDPs.actiontype(::MG{S,A}) where {S,A} = A

"""
    stateindex(game::POMG, s)
"""
function POMDPs.stateindex(::Game, s) end

"""
    player_actionindex(game::POMG, i::Int, a)
"""
function player_actionindex end

"""
    player_obs(game::POMG, i::Int, a)
"""
function player_obsindex end


##

"""
    solve(solver, game::Game)

Solves the Game using method associated with solver, and returns a policy.
"""
function POMDPs.solve(solver, game::Game) end
