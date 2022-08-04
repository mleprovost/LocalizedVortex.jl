using ProgressMeter
using VortexAssim

export smapassim

# Create a function to perform the sequential assimilation for any sequential filter SeqFilter
function smapassim(algo::SeqFilter, ens::EnsembleStateMeas{Nx, Ny, Ne},tspan::Tuple{S,S}, config::VortexParams, idxp, pressure_data; P::Parallel = serial) where {Nx, Ny, Ne, T<:Real, S<:Real}

# Define the additive Inflation
TransportMap.@get config (ϵX, ϵΓ, ϵLESP, β)
ϵx = RecipeInflation([ϵX; ϵΓ; ϵLESP])
ϵmul = MultiplicativeInflation(β)
tesp = config.tesp

h(t, x) = measure_state(vcat(x, tesp), t, config, idxp)
yt(t) = cfd_pressure(t, idxp, config, pressure_data);
dyn = DynamicalSystem(vortex, h)

t0, tf = tspan
step = ceil(Int, algo.Δtobs/algo.Δtdyn)
enshist = EnsembleState[]
push!(enshist, deepcopy(ens.state))

n0 = ceil(Int64, t0/algo.Δtobs) + 1
J = (tf-t0)/algo.Δtobs
Acycle = n0:n0+J#-1
@show length(Acycle)

# Run particle filter
@showprogress for i=1:length(Acycle)
	@show i
    # Forecast
	for j=1:step
		tj = t0+(i-1)*algo.Δtobs+(j-1)*algo.Δtdyn
		_, ens = vortex(tj, ens, config)
	end

    # Get real measurement
    ystar = yt(t0+i*algo.Δtobs)

	# Perform additive inflation for each ensemble member
	ϵmul(ens.state)
	if i>5
		ϵx(ens.state)
	end
	# Filter state
	if algo.isfiltered == true
		@inbounds for col in eachcol(ens.state.S)
			col .= algo.G(col)
		end
	end

	# Compute measurements
	observe(h, t0+(i-1)*algo.Δtobs, ens; P = P)
	ȳf = h(t0+(i-1)*algo.Δtobs, mean(ens.state))


	if i>10

	# Update the transport map
	dxx = dstate(ens.state, complex(t0+i*algo.Δtobs), config)
	dxy = dstatemeas(ens.state, complex(t0+i*algo.Δtobs), config, idxp)
	# @show size(dist)
	Nnew, _ = size(ens.state)
	@show Nnew
	p = 2
	diagobs_p = p
	off_p = p
	off_rad = 0.0#10*eps()
	nonid_rad = 1.0
	# order = [[-1], [p;p]]
	idx = zeros(Int64, 2,Ny)
	idx[1,:] .= collect(1:Ny)
	idx[2,:] .= collect(1:Ny)
	order = setup_vortexorder(diagobs_p, 0, off_p, off_rad, nonid_rad, dxx, dxy);
	# @show order



	algo = SparseTMap(Nnew, Ny, Ne, order, 2.0, 0.01, 1e-8, 10.0,  dxy, dyn, filter_state!, config.β,
                                               algo.ϵy, config.Δt, config.Δt, true, idx)

    # Generate posterior samples
	ens = algo(ens, ystar, t0 + i*algo.Δtobs)

	# Filter state
	if algo.isfiltered == true
		for col in eachcol(ens.state.S)
			col .= algo.G(col)
		end
	end

	end

    push!(enshist, deepcopy(ens.state))
end

return enshist
end
