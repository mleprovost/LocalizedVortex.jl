using Test

using LinearAlgebra, Statistics
using NlVortexAssim
using AdaptiveTransportMap
using PotentialFlow
import PotentialFlow.Elements: jacobian_position, jacobian_strength, jacobian_param

# include("differentialpressure.jl")
# include("factpressure.jl")
# include("projector.jl")
include("aggregation.jl")
include("finitediffpressure.jl")
# include("localization.jl")
