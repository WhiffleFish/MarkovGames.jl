@testset "games" begin
    game = CompetitiveTiger()
    b0 = initialstate(game)
    s = rand(b0)
    A1, A2 = actions(game)
    O1, O2 = observations(game)

    @test POMDPTools.has_consistent_distributions(game)
end

