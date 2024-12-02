module POMGs
# Maybe change to POSGs.jl ???

using Random
using POMDPs
import POMDPTools
using POMDPTools.POMDPDistributions
using POMDPTools.BeliefUpdaters
using Reexport
using SparseArrays

include("distributions.jl")
export ProductDistribution

include("pomg.jl")

include("gen_impl.jl")

include("sparse_tabular.jl")
export SparseTabularPOMG, SparseTabularMG, SparseTabularGame

include("consistency_check.jl")

include(joinpath("BeliefUpdaters", "BeliefUpdaters.jl"))
@reexport using .BeliefUpdaters

include(joinpath("games", "games.jl"))

end # module
