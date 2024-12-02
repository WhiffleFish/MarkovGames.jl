struct MatrixGame{N,T} <: POMG{Bool,NTuple{N,Int},NTuple{N,Nothing}}
    R::Array{NTuple{N,T}, N}
end

MatrixGame() = MatrixGame([
    (0,0) (-1,1) (1,-1);
    (1,-1) (0,0) (-1,1);
    (-1,1) (1,-1) (0,0)
])

POMDPs.updater(game::MatrixGame) = SingletonUpdater(game)

MarkovGames.initialstate(::MatrixGame) = Deterministic(false)

MarkovGames.players(::MatrixGame{N}) where N = 1:N

MarkovGames.isterminal(::MatrixGame, s::Bool) = s

MarkovGames.discount(::MatrixGame) = 1.0

MarkovGames.actions(g::MatrixGame) = axes(g.R)

MarkovGames.observation(::MatrixGame{N}, s::Bool, a::NTuple{N,Int}, sp::Bool) where N = Deterministic(NTuple{N, Nothing}(nothing for _ in 1:N))

MarkovGames.reward(g::MatrixGame{N}, s::Bool, a::NTuple{N,Int}) where N = g.R[a...]

MarkovGames.transition(::MatrixGame{N}, s::Bool, a::NTuple{N,Int}) where N = Deterministic(true)

function MarkovGames.gen(g::MatrixGame{N}, s::Bool, a::NTuple{N,Int}, rng) where N
    return (sp=true, o=NTuple{N, Nothing}(nothing for _ in 1:N), r=g.R[a...])
end
