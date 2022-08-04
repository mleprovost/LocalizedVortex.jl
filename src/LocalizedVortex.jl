module LocalizedVortex

using TransportBasedInference
using LinearAlgebra
using LoopVectorization
using PotentialFlow
using ProgressMeter

include("tools.jl")
include("aggregation.jl")
include("vortextools.jl")
include("state_equation.jl")
include("pressure.jl")
include("forecast.jl")
include("assimilation.jl")
include("localized_assimilation.jl")

end # module
