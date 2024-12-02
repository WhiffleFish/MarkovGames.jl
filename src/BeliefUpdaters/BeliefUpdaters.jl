module BeliefUpdaters

using ..POMGs
using Distributions
using POMDPTools.POMDPDistributions
using POMDPTools.BeliefUpdaters
using Random
using Statistics
using StatsBase
import POMDPs

include("discrete.jl")
export DiscretePOMGUpdater, DiscretePOMGBelief

include("singleton.jl")
export SingletonUpdater

end
