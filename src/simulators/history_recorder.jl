function Simulators.simulate(sim::HistoryRecorder,
    game::MG, 
    policy::Policy,
    s = rand(sim.rng, initialstate(game))
    )
    max_steps = something(sim.max_steps, typemax(Int))
    if !isnothing(sim.eps)
        max_steps = min(max_steps, ceil(Int,log(sim.eps)/log(discount(game))))
    end

    it = MGSimIterator(
        Simulators.default_spec(game),
        game,
        policy,
        sim.rng,
        s,
        max_steps
    )
    
    if sim.show_progress && isnothing(sim.max_steps) && isnothing(sim.eps)
        error("If show_progress=true in a HistoryRecorder, you must also specify max_steps or eps.")
    end

    prog = Progress(max_steps; desc="Simulating...", enabled=sim.show_progress)
    history, exception, backtrace = Simulators.collect_history(it, Val(sim.capture_exception), prog)
    finish!(prog)

    return SimHistory(Simulators.promote_history(history), discount(game), exception, backtrace)
end
