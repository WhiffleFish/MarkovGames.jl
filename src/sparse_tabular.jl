# assumed zero-sum
struct SparseTabularPOMG <: POMG{Int, Tuple{Int,Int}, Tuple{Int,Int}}
    T::Matrix{SparseMatrixCSC{Float64, Int64}} # T[a1, a2][sp, s]
    R::Array{Float64, 3} # R[s,a1,a2]
    O::NTuple{2,Matrix{SparseMatrixCSC{Float64, Int64}}} # O[i][a1, a2][sp, o] - only works with product distribution observations
    isterminal::SparseVector{Bool, Int}
    initialstate::SparseVector{Float64, Int}
    discount::Float64
end

struct SparseTabularMG <: MG{Int, Tuple{Int,Int}}
    T::Matrix{SparseMatrixCSC{Float64, Int64}} # T[a1, a2][sp, s]
    R::Array{Float64, 3} # R[s,a1,a2]
    isterminal::SparseVector{Bool, Int}
    initialstate::SparseVector{Float64, Int}
    discount::Float64
end

SparseTabularMG(game::SparseTabularMG) = game

function SparseTabularMG(game::Game)
    S = states(game)
    A = actions(game) # (A1, A2)

    terminal = _vectorized_terminal(game, S)
    T = _tabular_transitions(game, S, A, terminal)
    R = _tabular_rewards(game, S, A, terminal)
    b0 = _vectorized_initialstate(game, S)
    return SparseTabularMG(T,R,terminal,b0,discount(game))
end

SparseTabularMG(game::SparseTabularPOMG) = SparseTabularMG(game.T,game.R, game.isterminal, game.initialstate, game.discount)

function SparseTabularPOMG(game::POMG)
    S = states(game)
    A = actions(game) # (A1, A2)
    O = observations(game) # (O1, O2)

    terminal = _vectorized_terminal(game, S)
    T = _tabular_transitions(game, S, A, terminal)
    R = _tabular_rewards(game, S, A, terminal)
    O = _tabular_observations(game, S, A, O)
    b0 = _vectorized_initialstate(game, S)
    return SparseTabularPOMG(T,R,O,terminal,b0,discount(game))
end

SparseTabularPOMG(game::SparseTabularPOMG) = game

function _tabular_transitions(game::POMG, S, A, terminal)
    ns = length(S)
    na1, na2 = length.(A)
    T = [zeros(ns,ns) for i ∈ 1:na1, j ∈ 1:na2]
    for idx ∈ CartesianIndices(T)
        a_idxs = Tuple(idx)
        a = first(A)[first(a_idxs)], last(A)[last(a_idxs)]
        _fill_transitions!(game, T[idx], S, a, terminal)
    end
    T
end

function _fill_transitions!(game::POMG, T, S, a, terminal)
    for (s_idx, s) ∈ enumerate(S)
        if terminal[s_idx]
            T[:, s_idx] .= 0.0
            T[s_idx, s_idx] = 1.0
            continue
        end
        Tsa = transition(game, s, a)
        for (sp_idx, sp) ∈ enumerate(S)
            T[sp_idx, s_idx] = POMDPs.pdf(Tsa, sp)
        end
    end
    T
end

function _tabular_rewards(game, S, A, terminal)
    A1, A2 = A
    R = Array{Float64}(undef, length(S), length(A1), length(A2))
    for (s_idx, s) ∈ enumerate(S)
        if terminal[s_idx]
            R[s_idx, :, :] .= 0.0
            continue
        end
        for (a1_idx, a1) ∈ enumerate(A1)
            for (a2_idx, a2) ∈ enumerate(A2)
                R[s_idx, a1_idx, a2_idx] = first(reward(game, s, (a1, a2))) # only recording reward for player 1 - assumed zero-sum
            end
        end
    end
    R
end

function _tabular_observations(game, S, A, O)
    A1,A2 = A
    O1,O2 = O
    _O = (
        [Matrix{Float64}(undef, length(S), length(O1)) for _ ∈ eachindex(A1), _ ∈ eachindex(A2)], 
        [Matrix{Float64}(undef, length(S), length(O2)) for _ ∈ eachindex(A1), _ ∈ eachindex(A2)]
    )
    for i ∈ 1:2
        for (a1_idx, a1) ∈ enumerate(A1), (a2_idx, a2) ∈ enumerate(A2)
            _fill_observations!(game, i, _O[i][a1_idx, a2_idx], S, (a1, a2), O[i])
        end
    end
    _O
end

function _fill_observations!(game, i, Oa, S, a, O)
    for (sp_idx, sp) ∈ enumerate(S)
        obs_dist = player_observation(game, i, a, sp)
        for (o_idx, o) ∈ enumerate(O)
            Oa[sp_idx, o_idx] = POMDPs.pdf(obs_dist, o)
        end
    end
    Oa
end

function _vectorized_terminal(game, S)
    term = BitVector(undef, length(S))
    @inbounds for i ∈ eachindex(S)
        term[i] = isterminal(game, S[i])
    end
    return term
end

function _vectorized_initialstate(game, S)
    b0 = initialstate(game)
    b0_vec = Vector{Float64}(undef, length(S))
    @inbounds for i ∈ eachindex(S)
        b0_vec[i] = POMDPs.pdf(b0, S[i])
    end
    return sparse(b0_vec)
end

const SparseTabularGame = Union{SparseTabularPOMG, SparseTabularMG}

POMDPTools.ordered_states(game::SparseTabularGame) = axes(game.R, 1)
POMDPs.states(game::SparseTabularGame) = axes(game.R, 1)
POMDPTools.ordered_actions(game::SparseTabularGame) = axes(game.T)
POMDPs.actions(game::SparseTabularGame) = axes(game.T)
POMDPTools.ordered_observations(game::SparseTabularPOMG) = axes(first(game.O), 2)
POMDPs.observations(game::SparseTabularPOMG) = axes(first(game.O), 2)

POMDPs.discount(game::SparseTabularGame) = game.discount
POMDPs.initialstate(game::SparseTabularGame) = SparseCat(states(game), game.initialstate)
POMDPs.isterminal(game::SparseTabularGame, s::Int) = game.isterminal[s]

function POMDPs.transition(game::SparseTabularGame, s::Int, a)
    T_a = game.T[a...]
    Tnz = nonzeros(T_a)
    Trv = rowvals(T_a)
    sp_idxs = nzrange(T_a, s)
    sps = @view Trv[sp_idxs]
    probs = @view Tnz[sp_idxs]
    return SparseCat(sps, probs)
end

POMGs.player_observation(game::SparseTabularPOMG, i, a, sp) = SparseCat(
    observations(game)[i], 
    game.O[i][a...][sp,:]
)


n_states(game::SparseTabularGame) = length(states(game))
n_actions(game::SparseTabularGame) = length.(actions(game))
n_observations(game::SparseTabularPOMG) = length.(observations(game))
