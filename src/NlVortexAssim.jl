module NlVortexAssim

using TransportBasedInference
using FiniteDiff
using LinearAlgebra
using LoopVectorization
using LowRankApprox
using PotentialFlow
using ProgressMeter
using OrthoMatchingPursuit
using Statistics
using ThreadTools

include("tools.jl")
include("aggregation.jl")
include("vortextools.jl")
include("state_equation.jl")
include("pressure.jl")
include("ADpressure.jl")
include("forecast.jl")
include("finitediff.jl")
include("finitediffpressure.jl")
include("sparsity.jl")
include("assimilation.jl")
include("lowrank.jl")
include("localized_assimilation.jl")
include("source.jl")

end # module
