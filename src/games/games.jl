module Games

using ..POMGs
using POMDPTools.POMDPDistributions
using StaticArrays
using Base.Iterators
using Random
import POMDPs

include("matrix.jl")
export MatrixGame

include("kuhn.jl")
export Kuhn

include("tiger.jl")
export CompetitiveTiger

end
