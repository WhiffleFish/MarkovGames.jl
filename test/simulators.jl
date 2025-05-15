@testset "Simulators" begin
    game = SimpleMG()
    sol = DummySolver()
    pol = solve(sol, game)
    
    ## History
    sim = HistoryRecorder(max_steps=10)
    hist = simulate(sim, game, pol)
    @test all(iszero, hist[:r])
    @test length(hist) == 10
    @test all(hist[:behavior]) do σ
        σ == ProductDistribution(Deterministic.(first.(actions(game))))
    end
    @test all(isone, hist[:s])
    @test all(==((1,1)), hist[:a])
    @test all(isone, hist[:sp])


    ## Rollouts
    sim = RolloutSimulator(max_steps=10)
    ret = simulate(sim, game, pol, rand(initialstate(game)))
    @assert ret isa Float64
end
