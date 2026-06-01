struct InfoPolicy{G} <: Policy
    game::G
end

MarkovGames.behavior(p::InfoPolicy, s) = ProductDistribution(
    Deterministic.(first.(actions(p.game)))
)

MarkovGames.behavior_info(p::InfoPolicy, s) = (
    behavior(p, s),
    (state_marker=s,)
)

mutable struct BehaviorInfoTotal <: RolloutStat
    total::Int
end

BehaviorInfoTotal() = BehaviorInfoTotal(0)

function MarkovGames.reset!(stat::BehaviorInfoTotal)
    stat.total = 0
    return stat
end

function MarkovGames.observe_step!(stat::BehaviorInfoTotal, game::SimpleMG, step::RolloutStep)
    stat.total += step.behavior_info.state_marker
    return stat
end

MarkovGames.stat_result(stat::BehaviorInfoTotal) = (behavior_info_total=stat.total,)

mutable struct StartedAtTwo <: RolloutStat
    reached::Bool
end

StartedAtTwo() = StartedAtTwo(false)

function MarkovGames.reset!(stat::StartedAtTwo)
    stat.reached = false
    return stat
end

function MarkovGames.observe_step!(stat::StartedAtTwo, game::SimpleMG, step::RolloutStep)
    if step.t == 1 && step.s == 2
        stat.reached = true
    end
    return stat
end

MarkovGames.stat_result(stat::StartedAtTwo) = (evader_reached_goal=stat.reached,)

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
    @test ret isa SVector{2,Float64}

    ## Stat Rollouts
    stat_sim = StatRolloutSimulator(max_steps=10, accumulators=(StepCount(),))
    stat_ret = simulate(stat_sim, game, pol, rand(initialstate(game)))
    @test stat_ret.reward isa SVector{2,Float64}
    @test stat_ret.steps == 10
    @inferred simulate(stat_sim, game, pol, rand(initialstate(game)))
    @inferred MarkovGames.stat_result((StepCount(),))

    empty_stat_sim = StatRolloutSimulator(max_steps=10)
    empty_stat_ret = simulate(empty_stat_sim, game, pol, rand(initialstate(game)))
    @test keys(empty_stat_ret) == (:reward,)
    @inferred simulate(empty_stat_sim, game, pol, rand(initialstate(game)))
    @inferred MarkovGames.stat_result(())

    info_pol = InfoPolicy(game)
    info_sim = StatRolloutSimulator(max_steps=3, accumulators=(BehaviorInfoTotal(),))
    info_ret = simulate(info_sim, game, info_pol, rand(initialstate(game)))
    @test info_ret.behavior_info_total == 3
    @inferred simulate(info_sim, game, info_pol, rand(initialstate(game)))

    batch_ret = run_stats_parallel(
        game,
        pol,
        4;
        accumulators=(StepCount(), StartedAtTwo()),
        batch_accumulators=(
            MeanResult(:steps; name=:mean_steps),
            RateResult(:evader_reached_goal; name=:evader_goal_rate),
        ),
        initialstates=[1, 2, 2, 1],
        max_steps=4,
        show_progress=false,
        proc_warn=false,
    )
    @test batch_ret.reward == SA[0.5, -0.5]
    @test batch_ret.mean_steps == 4.0
    @test batch_ret.evader_goal_rate == 0.5

    mean_steps = MeanResult(:steps; name=:mean_steps)
    evader_goal_rate = RateResult(:evader_reached_goal; name=:evader_goal_rate)
    sim_result = (reward=SA[0.0, 0.0], steps=4, evader_reached_goal=true)
    @inferred MarkovGames.observe_sim!(mean_steps, sim_result)
    @inferred MarkovGames.observe_sim!(evader_goal_rate, sim_result)
    @inferred MarkovGames.batch_result((mean_steps, evader_goal_rate))
    @inferred run_stats_parallel(
        game,
        pol,
        2;
        accumulators=(StepCount(), StartedAtTwo()),
        batch_accumulators=(
            MeanResult(:steps; name=:mean_steps),
            RateResult(:evader_reached_goal; name=:evader_goal_rate),
        ),
        initialstates=[1, 2],
        max_steps=2,
        show_progress=false,
        proc_warn=false,
    )

    reward_ret = run_stats_parallel(
        game,
        pol,
        2;
        initialstates=[1, 2],
        max_steps=2,
        show_progress=false,
        proc_warn=false,
    )
    @test keys(reward_ret) == (:reward,)
    @test reward_ret.reward == SA[0.5, -0.5]
end
