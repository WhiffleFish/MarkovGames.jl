using MarkovGames
using MarkovGames.Games
using POMDPTools
using POMDPs
using Test

@testset "generative" begin
    game = MatrixGame()
    b = initialstate(game)
    @test b isa Deterministic
    s = b.val
    a1 = first(player_actions(game, 1, s))
    a2 = last(player_actions(game, 2, s))
    a = (a1, a2)

    sp, (o1,o2), (r1, r2) = @gen(:sp, :o, :r)(game, s, a)
    @test sp isa Bool
    @test o1 == o2 == nothing
    @test r1 == -r2
    @test isterminal(game, sp)
end

include("games.jl")

include("distributions.jl")

struct SimpleMG <: MG{Int, Tuple{Int,Int}} end

MarkovGames.states(::SimpleMG) = (1,2)
MarkovGames.actions(::SimpleMG) = ((1,2), (1,2))
MarkovGames.discount(::SimpleMG) = 1.0
MarkovGames.transition(::SimpleMG, s, (a1, a2)) = a1 == a2 ? Deterministic(1) : Deterministic(2)
MarkovGames.reward(::SimpleMG, s, a) = isone(s) ? 0.0 : 1.0
MarkovGames.stateindex(::SimpleMG, s) = s
MarkovGames.actionindex(::SimpleMG, a) = a
MarkovGames.initialstate(::SimpleMG) = Deterministic(1)

struct DummySolver end

struct DummyPolicy{G} <: Policy
    game::G
end
MarkovGames.solve(::DummySolver, game::MG) = DummyPolicy(game)
MarkovGames.behavior(p::DummyPolicy, s) = ProductDistribution(
    Deterministic.(first.(actions(p.game)))
)

include("simulators.jl")

include("sparse_tabular.jl")

include("exploitability.jl")
