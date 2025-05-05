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

    ##
    @testset "SimpleMG" begin
        game = SimpleMG()
        sparse_game = SparseTabularMG(game)
        for s ∈ states(game)
            for a ∈ actions(game)
                @test only(support(transition(game, s, a))) == only(support(transition(sparse_game, s, a)))
            end
        end
    end
end
