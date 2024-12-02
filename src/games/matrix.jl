struct MatrixGame{N,T} <: POMG{Bool,NTuple{N,Int},NTuple{N,Nothing}}
    R::Array{NTuple{N,T}, N}
end

MatrixGame() = MatrixGame([
    (0,0) (-1,1) (1,-1);
    (1,-1) (0,0) (-1,1);
    (-1,1) (1,-1) (0,0)
])

POMDPs.updater(game::MatrixGame) = SingletonUpdater(game)

POMGs.initialstate(::MatrixGame) = Deterministic(false)

POMGs.players(::MatrixGame{N}) where N = 1:N

POMGs.isterminal(::MatrixGame, s::Bool) = s

POMGs.discount(::MatrixGame) = 1.0

POMGs.actions(g::MatrixGame) = axes(g.R)

POMGs.observation(::MatrixGame{N}, s::Bool, a::NTuple{N,Int}, sp::Bool) where N = Deterministic(NTuple{N, Nothing}(nothing for _ in 1:N))

POMGs.reward(g::MatrixGame{N}, s::Bool, a::NTuple{N,Int}) where N = g.R[a...]

POMGs.transition(::MatrixGame{N}, s::Bool, a::NTuple{N,Int}) where N = Deterministic(true)

function POMGs.gen(g::MatrixGame{N}, s::Bool, a::NTuple{N,Int}, rng) where N
    return (sp=true, o=NTuple{N, Nothing}(nothing for _ in 1:N), r=g.R[a...])
end
