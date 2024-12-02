using POMGs
using POMGs.Games
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

@testset "sparse tabular" begin
    tiger = CompetitiveTiger()
    S = states(tiger)
    A1, A2 = actions(tiger)
    O1, O2 = observations(tiger)
    game = SparseTabularPOMG(tiger)
    @test all(game.T) do T
        size(T) == (length(S), length(S))
    end
    @test size(game.O[1]) == size(game.O[2]) == (length(A1), length(A2))
    @test all(size(O_a) == (length(S), length(O1)) for O_a ∈ game.O[1])
    @test all(game.O[1]) do O_a
        all(eachrow(O_a)) do po_a_sp
            isone(sum(po_a_sp))
        end
    end
    @test all(size(O_a) == (length(S), length(O2)) for O_a ∈ game.O[2])
    @test all(game.O[2]) do O_a
        all(eachrow(O_a)) do po_a_sp
            isone(sum(po_a_sp))
        end
    end
    @test size(game.T) == (length(A1), length(A2))
    @test all(game.T) do T_a
        all(eachcol(T_a)) do Tsa
            isone(sum(Tsa))
        end
    end
end


