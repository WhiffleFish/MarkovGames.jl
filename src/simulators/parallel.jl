struct MGSim{SIM<:Simulator, M<:MG, P<:Policy, S, NT<:NamedTuple} <: Sim
    simulator::SIM
    mg::M
    policy::P
    initialstate::S
    metadata::NT
end

Simulators.problem(sim::MGSim) = sim.mg

"""
    Sim(m::MG, p::Policy, metadata=(note="a note",))
    Sim(m::MG, p::Policy[, initialstate]; kwargs...)

Create a `Sim` object that represents a MDP simulation.
"""
function Simulators.Sim(mg::MG,
        policy::Policy,
        is                  = rand(rng, initialstate(mg));
        rng::AbstractRNG    = Random.default_rng(),
        max_steps::Int      = typemax(Int),
        simulator::Simulator= HistoryRecorder(rng=rng, max_steps=max_steps),
        metadata            = NamedTuple()
    )
    return MGSim(simulator, mg, policy, is, merge(NamedTuple(), metadata))
end

POMDPs.simulate(s::MGSim) = simulate(s.simulator, s.mg, s.policy, s.initialstate)
