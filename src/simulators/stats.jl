export
    StatRolloutSimulator,
    RolloutStep,
    RolloutStat,
    BatchStat,
    reset_stat!,
    observe_step!,
    stat_result,
    reset_batch!,
    observe_sim!,
    batch_result,
    StepCount,
    MeanResult,
    RateResult,
    run_stats_parallel

abstract type RolloutStat end
abstract type BatchStat end

"""
    RolloutStep

Data passed to each `RolloutStat` after one simulated transition.

Fields are `t`, `s`, `a`, `sp`, `r`, `info`, `behavior`, and `behavior_info`.
"""
struct RolloutStep{S,A,R,I,B,BI}
    t::Int
    s::S
    a::A
    sp::S
    r::R
    info::I
    behavior::B
    behavior_info::BI
end

"""
    StatRolloutSimulator(; rng=Random.default_rng(), eps=0.0, max_steps=typemax(Int), accumulators=())

Run a fully observable `MG` rollout and collect mutable per-rollout statistics.

`accumulators` is a tuple of `RolloutStat` prototypes. The simulator deep-copies
and resets those prototypes at the start of each simulation, calls
`observe_step!` on every step, and returns a `NamedTuple` containing `reward`
merged with all `stat_result` outputs.
"""
struct StatRolloutSimulator{RNG<:AbstractRNG, A<:Tuple} <: Simulator
    rng::RNG
    max_steps::Int
    eps::Float64
    accumulators::A
end

function StatRolloutSimulator(;
    rng::AbstractRNG=Random.default_rng(),
    eps::Float64=0.0,
    max_steps::Int=typemax(Int),
    accumulators=()
)
    return StatRolloutSimulator(rng, max_steps, eps, _as_tuple(accumulators))
end

_as_tuple(x::Tuple) = x
_as_tuple(x::AbstractVector) = Tuple(x)
_as_tuple(x) = (x,)

_named_result(::Val{name}, value) where {name} = NamedTuple{(name,)}((value,))

"""
    reset_stat!(stat::RolloutStat)

Reset a per-rollout statistic before a simulation starts.
"""
reset_stat!(stat::RolloutStat) = stat

"""
    observe_step!(stat::RolloutStat, game::MG, step::RolloutStep)

Update a per-rollout statistic from one simulated transition.
"""
observe_step!(stat::RolloutStat, game::MG, step::RolloutStep) = stat

"""
    stat_result(stat::RolloutStat)

Return a `NamedTuple` containing this statistic's per-simulation result.
"""
stat_result(stat::RolloutStat) = NamedTuple()
stat_result(::Tuple{}) = NamedTuple()
stat_result(stats::Tuple) = merge(stat_result(first(stats)), stat_result(Base.tail(stats)))

function _fresh_stats(accumulators::Tuple)
    stats = deepcopy(accumulators)
    foreach(reset_stat!, stats)
    return stats
end

function _observe_stats!(accumulators::Tuple, game::MG, step::RolloutStep)
    foreach(acc -> observe_step!(acc, game, step), accumulators)
    return accumulators
end

mutable struct StepCount <: RolloutStat
    n::Int
end

"""
    StepCount()

Per-rollout statistic that returns `(steps=n,)`.
"""
StepCount() = StepCount(0)

function reset_stat!(stat::StepCount)
    stat.n = 0
    return stat
end

function observe_step!(stat::StepCount, game::MG, step::RolloutStep)
    stat.n += 1
    return stat
end

stat_result(stat::StepCount) = (steps=stat.n,)

function POMDPs.simulate(
    sim::StatRolloutSimulator,
    game::MG{S},
    policy::Policy,
    s::S;
    rt::Type{RT}=reward_type(game)
) where {S,RT}
    (;rng, max_steps) = sim
    accumulators = _fresh_stats(sim.accumulators)

    gamma_t = 1.0
    gamma = discount(game)
    r_total = zero(RT)
    step = 1

    while gamma_t > sim.eps && !isterminal(game, s) && step <= max_steps
        behavior_dist, behavior_meta = behavior_info(policy, s)
        a = rand(rng, behavior_dist)
        sp, r, info = @gen(:sp,:r,:info)(game, s, a, rng)
        rollout_step = RolloutStep(step, s, a, sp, r, info, behavior_dist, behavior_meta)
        _observe_stats!(accumulators, game, rollout_step)

        r_total = r_total .+ gamma_t .* r
        s = sp
        gamma_t *= gamma
        step += 1
    end

    return merge((reward=r_total,), stat_result(accumulators))
end

"""
    reset_batch!(stat::BatchStat)

Reset a batch statistic before aggregating simulation results.
"""
reset_batch!(stat::BatchStat) = stat

"""
    observe_sim!(stat::BatchStat, result)

Update a batch statistic from one per-simulation result `NamedTuple`.
"""
observe_sim!(stat::BatchStat, result) = stat

"""
    batch_result(stat::BatchStat)

Return a `NamedTuple` containing this statistic's aggregate batch result.
"""
batch_result(stat::BatchStat) = NamedTuple()
batch_result(::Tuple{}) = NamedTuple()
batch_result(stats::Tuple) = merge(batch_result(first(stats)), batch_result(Base.tail(stats)))

function _fresh_batch_stats(accumulators::Tuple)
    stats = deepcopy(accumulators)
    foreach(reset_batch!, stats)
    return stats
end

function _observe_batch_stats!(accumulators::Tuple, result)
    foreach(acc -> observe_sim!(acc, result), accumulators)
    return accumulators
end

mutable struct MeanResult{K,N} <: BatchStat
    total::Float64
    count::Int
end

"""
    MeanResult(key::Symbol; name::Symbol=Symbol(:mean_, key))

Batch statistic that averages a numeric field from each per-simulation result.
"""
MeanResult(key::Symbol; name::Symbol=Symbol(:mean_, key)) = MeanResult{key,name}(0.0, 0)

function reset_batch!(stat::MeanResult)
    stat.total = 0.0
    stat.count = 0
    return stat
end

function observe_sim!(stat::MeanResult{K}, result) where {K}
    stat.total += float(result[K])
    stat.count += 1
    return stat
end

function batch_result(stat::MeanResult{K,N}) where {K,N}
    value = stat.count == 0 ? NaN : stat.total / stat.count
    return _named_result(Val(N), value)
end

mutable struct RateResult{K,N} <: BatchStat
    successes::Int
    count::Int
end

"""
    RateResult(key::Symbol; name::Symbol=Symbol(key, :_rate))

Batch statistic that computes the fraction of per-simulation boolean results
where `key` is true.
"""
RateResult(key::Symbol; name::Symbol=Symbol(key, :_rate)) = RateResult{key,name}(0, 0)

function reset_batch!(stat::RateResult)
    stat.successes = 0
    stat.count = 0
    return stat
end

function observe_sim!(stat::RateResult{K}, result) where {K}
    stat.successes += result[K] ? 1 : 0
    stat.count += 1
    return stat
end

function batch_result(stat::RateResult{K,N}) where {K,N}
    value = stat.count == 0 ? NaN : stat.successes / stat.count
    return _named_result(Val(N), value)
end

"""
    run_stats_parallel(game, policy, n; kwargs...)

Run `n` `StatRolloutSimulator` simulations through POMDPTools `run_parallel`
and aggregate their per-simulation results with `batch_accumulators`.

Common keyword arguments are `accumulators`, `batch_accumulators`, `rng`,
`max_steps`, `eps`, `initialstates`, `metadata`, `pool`, `show_progress`, and
`proc_warn`. If `pool` is omitted, the default `run_parallel` worker pool is
used. Returns one aggregate `NamedTuple`.
"""
function run_stats_parallel(
    game::MG,
    policy::Policy,
    n::Integer;
    accumulators=(),
    batch_accumulators=(),
    max_steps::Int=typemax(Int),
    eps::Float64=0.0,
    initialstates=nothing,
    metadata=NamedTuple(),
    pool=nothing,
    show_progress::Bool=true,
    proc_warn::Bool=false
)
    rollout_accumulators = _as_tuple(accumulators)
    batch_stats = _fresh_batch_stats(_as_tuple(batch_accumulators))
    queue = Vector{Sim}(undef, n)

    for i in 1:n
        sim_rng = Random.default_rng()
        initial_state = isnothing(initialstates) ? rand(sim_rng, initialstate(game)) : initialstates[i]
        simulator = StatRolloutSimulator(
            rng=sim_rng,
            eps=eps,
            max_steps=max_steps,
            accumulators=rollout_accumulators
        )
        queue[i] = Sim(game, policy, initial_state; rng=sim_rng, simulator=simulator, metadata=metadata)
    end

    process = (sim, result) -> result
    rows = if isnothing(pool)
        run_parallel(process, queue; show_progress=show_progress, proc_warn=proc_warn)
    else
        run_parallel(process, queue, pool; show_progress=show_progress, proc_warn=proc_warn)
    end

    for row in eachrow(rows)
        _observe_batch_stats!(batch_stats, NamedTuple(row))
    end

    return batch_result(batch_stats)
end
