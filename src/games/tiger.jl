@enum TigerState TIGER_LEFT=1 TIGER_RIGHT=2 NOTHING=3
@enum TigerAction BLOCK=1 LISTEN=2 OPEN_LEFT=3 OPEN_RIGHT=4

Base.@kwdef struct CompetitiveTiger <: POMG{TigerState, Tuple{TigerAction,TigerAction}, Tuple{TigerState,TigerState}}
    r_listen::Float64           = -1.0
    r_findtiger::Float64        = -100.
    r_escapetiger::Float64      = 10.
    p_listen_correctly::Float64 = 0.85
    discount::Float64           = 0.95
end

const R_TIGER = let
    # A1 × S × A2
    _R = Float64[
    [
        0 0
        -1 -1
        -1 1
        1 -1
    ];;;
    [
        1 1
        0 0
        4 -2
        -2 4
    ];;;
    [
        1 -1
        -4 2
        0 0
        -6 6
    ];;;
    [
        -1 1
        2 -4
        6 -6
        0 0
    ]]
    # convert to S × A1 × A2
    SArray{Tuple{2,4,4}}(permutedims(_R, (2,1,3)))
end

POMGs.discount(game::CompetitiveTiger) = game.discount
POMGs.initialstate(::CompetitiveTiger) = Uniform((TIGER_LEFT, TIGER_RIGHT))

POMGs.states(::CompetitiveTiger) = (TIGER_LEFT, TIGER_RIGHT)
POMGs.actions(::CompetitiveTiger) = (instances(TigerAction), instances(TigerAction))
POMGs.observations(::CompetitiveTiger) = (instances(TigerState), instances(TigerState))

POMGs.stateindex(::CompetitiveTiger, s::TigerState) = Int(s)
POMGs.player_actionindex(::CompetitiveTiger, i::Int, a::TigerAction) = Int(a)
POMGs.player_obsindex(::CompetitiveTiger, i::Int, o::TigerState) = Int(o)

function POMGs.transition(::CompetitiveTiger, s::TigerState, a::Tuple{TigerAction,TigerAction})
    if a == OPEN_LEFT || a == OPEN_RIGHT
        p = 0.5
    elseif s == TIGER_RIGHT
        p = 1.0
    else
        p = 0.0
    end
    return SparseCat(SA[TIGER_LEFT, TIGER_RIGHT], SA[1-p, p])
end

function POMGs.reward(::CompetitiveTiger, s::TigerState, a::Tuple{TigerAction,TigerAction})
    p1_reward = R_TIGER[Int(s), Int.(a)...]
    return (p1_reward, -p1_reward)
end

function POMGs.observation(game::CompetitiveTiger, a, sp)
    return ProductDistribution(
        player_observation(game, 1, a, sp), 
        player_observation(game, 2, a, sp)
    )
end

# FIXME: Not type stable
# TODO: not using `p_listen_correctly` game field
function POMGs.player_observation(::CompetitiveTiger, p::Int, a::Tuple{TigerAction,TigerAction}, sp::TigerState)
    return if a[p] == LISTEN
        if sp == TIGER_LEFT
            SparseCat(SA[TIGER_LEFT, TIGER_RIGHT, NOTHING], SA[0.85, 0.15, 0.0])
        else
            SparseCat(SA[TIGER_LEFT, TIGER_RIGHT, NOTHING], SA[0.15, 0.85, 0.0])
        end
    else
        SparseCat(SA[TIGER_LEFT, TIGER_RIGHT, NOTHING], SA[0.0, 0.0, 1.0])
    end
end
