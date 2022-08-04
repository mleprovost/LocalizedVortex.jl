
export vortex, vortex_mean, new_vortex, vortex2

import TransportBasedInference: Parallel, Serial, Thread

# The aggregation procedure is now common, we do the transfer that on commonly induced the lowest error, should we look at the norm Inf?

vortex2(X, t, Ny, Nx, config::VortexParams) = vortex2(X, t, Ny, Nx, config, serial)

# ; ϵaggregation1::Float64 = 3e-4, ϵaggregation2::Float64 = 8e-4)
function vortex2(X, t::Float64, Ny, Nx, config::VortexParams, P::Serial; ϵaggregation1::Float64 = 3e-4, ϵaggregation2::Float64 = 1.5e-3)
	Nypx, Ne = size(X)
	@assert Nypx == Ny + Nx "Wrong value of Ny or Nx"
	tmp = Array{Float64,1}[]
	empty_inds = Vector{Int}[]
    len = Int[]
	tesp = config.tesp

	# @show ceil(Int64, (Nx-1)/3)

	if config.transfer && ceil(Int64, (Nx-1)÷3) > 20-2
		# @show "aggregation"
		cache = Array{Float64,1}[]
		ΔZ = Array{ComplexF64,2}[]
		err = Array{ComplexF64,2}[]
	    @inbounds for i = 1:Ne
			col = view(X, Ny+1:Nypx, i)
	        new_state, ΔZi, erri = state_equation2(vcat(col, tesp), t, config)
	        push!(cache, new_state)
			push!(ΔZ, ΔZi)
			push!(err, erri)
	    end

		err_mean = zeros(Float64, size(err[1]))

		for i=1:Ne
			err_mean += abs.(err[i])
		end
		rmul!(err_mean, 1/Ne)

		# Add another constraint on the size
		plate = Plate(config.N, config.L, complex(t+config.Δt), config.α)

		inds = collect(Iterators.filter(i -> abs(err_mean[i]) < ϵaggregation1, CartesianIndices(size(err_mean))))
		@inbounds for i = 1:Ne
			tmp_blobs, lesp, tesp = state_to_blobs(cache[i], config.δ)
			transfer_circulation!(tmp_blobs, plate, ΔZ[i], err_mean, inds, ϵaggregation2)
			push!(empty_inds, findall(b -> circulation(b) == 0, tmp_blobs))
			push!(len, length(tmp_blobs))
			push!(tmp, blobs_to_state(tmp_blobs, lesp, tesp))
		end
	else
		# @show "no aggregation"
		@inbounds for i = 1:Ne
			col = view(X, Ny+1:Nypx, i)
			new_state = state_equation2(vcat(col, tesp), t, config)
			# new_state = state_equation(states[i], t, config)
			new_blobs, lesp, tesp = state_to_blobs(new_state, config.δ)
			push!(empty_inds, findall(b -> circulation(b) == 0, new_blobs))
			push!(len, length(new_blobs))
			push!(tmp, new_state)
		end
	end
    # trim zeros
    @assert all(len .== len[1])
    toremove = intersect(empty_inds...)
    tokeep = filter(i -> i ∉ toremove, 1:len[1])

	tmpfinal = Array{Float64,1}[]
    for i=1:Ne
        new_blobs, lesp, tesp = state_to_blobs(tmp[i], config.δ)
        push!(tmpfinal, blobs_to_state(new_blobs[tokeep], lesp, tesp)[1:end-1])
    end

	if Ne==1
		X = vcat(X[1:Ny,:], reshape(tmpfinal[1], (size(tmpfinal[1], 1), 1)))
    else
		X = vcat(X[1:Ny,:], hcat(tmpfinal...))
    end
		return X, t + config.Δt
end

# In this version of vortex, the measurement and state vectors are stored together
# The first Ny components store the measurements, the next Nx components store
# the state

vortex(X, t, Ny, Nx, config::VortexParams) = vortex(X, t, Ny, Nx, config, serial)

function vortex(X, t::Float64, Ny, Nx, config::VortexParams, P::Serial)
	Nypx, Ne = size(X)
	@assert Nypx == Ny + Nx "Wrong value of Ny or Nx"
    tmp = Array{Float64,1}[]
    empty_inds = Vector{Int}[]
    len = Int[]
	tesp = config.tesp

    @inbounds for i = 1:Ne
		col = view(X, Ny+1:Nypx, i)
        new_state = state_equation(vcat(col, tesp), t, config)
		# new_state = state_equation(states[i], t, config)
        new_blobs, lesp, tesp = state_to_blobs(new_state, config.δ)
        push!(empty_inds, findall(b -> circulation(b) == 0, new_blobs))
        push!(len, length(new_blobs))
        push!(tmp, new_state)
    end

    # trim zeros
    @assert all(len .== len[1])
    toremove = intersect(empty_inds...)
    tokeep = filter(i -> i ∉ toremove, 1:len[1])

    @inbounds for (i, state) in enumerate(tmp)
        new_blobs, lesp, tesp = state_to_blobs(state, config.δ)
        tmp[i] = blobs_to_state(new_blobs[tokeep], lesp, tesp)[1:end-1]
    end

	if Ne==1
		X = vcat(X[1:Ny,:], reshape(tmp[1], (size(tmp[1], 1), 1)))
    else
		X = vcat(X[1:Ny,:], hcat(tmp...))
    end
		return X, t + config.Δt
end


function vortex(X, t::Float64, Ny, Nx, config::VortexParams, P::Thread)
	Nypx, Ne = size(X)
	@assert Nypx == Ny + Nx "Wrong value of Ny or Nx"
    tmp = Array{Float64,1}[]
    empty_inds = Vector{Int}[]
    len = Int[]
	tesp = config.tesp

	# Multi-threaded map with ordered result, provided by ThreadTools.jl
	tmp = tmap(x-> begin; state_equation(vcat(x, tesp), t, config); end, eachcol(view(X, Ny+1:Nypx,:)))

	@inbounds for i=1:Ne
		push!(empty_inds, findall(Γi -> Γi == 0, tmp[i][3:3:end]))
		push!(len, ceil(Int64, (size(tmp[i],1)-2)/3))
	end

    # trim zeros
    @assert all(len .== len[1])
    toremove = intersect(empty_inds...)
    tokeep = filter(i -> i ∉ toremove, 1:len[1])

    @inbounds for (i, state) in enumerate(tmp)
        new_blobs, lesp, tesp = state_to_blobs(state, config.δ)
        tmp[i] = blobs_to_state(new_blobs[tokeep], lesp, tesp)[1:end-1]
    end
	# The dimension of X is modified with the newly shed and aggregated vortex elements
	Nypx = Ny + size(tmp[1],1)


    if Ne==1
		X = vcat(X[1:Ny,:], reshape(tmp[1], (size(tmp[1], 1), 1)))
    else
		X = vcat(X[1:Ny,:], hcat(tmp...))
    end
		return X, t + config.Δt
end



vortex_mean(X, t::Float64, Ny, Nx, config::VortexParams) = vortex_mean(X, t, Ny, Nx, config, serial)


function vortex_mean(X, t::Float64, Ny, Nx, config::VortexParams, P::Serial)
	Nypx, Ne = size(X)
	@assert Nypx == Ny + Nx "Wrong value of Ny or Nx"
    tmp = Array{Float64,1}[]
    empty_inds = Vector{Int}[]
    len = Int[]
	tesp = config.tesp

	# Perform the aggregation on the mean to determine the set of vortices to aggregate
	X̄ = mean(view(X, Ny+1:Nypx, :), dims = 2)[:,1]
	new_mean, inds = mean_state_equation(vcat(X̄, tesp), t, config)
	new_blobsmean, _, _ = state_to_blobs(new_mean, config.δ)

	@inbounds for i = 1:Ne
		col = view(X, Ny+1:Nypx, i)
        new_state = state_equation(vcat(col, tesp), t, config, ϵaggregation1 = +Inf, ϵaggregation2=+Inf, inds = inds)
		# new_state = state_equation(states[i], t, config)
        new_blobs, lesp, tesp = state_to_blobs(new_state, config.δ)
        push!(empty_inds, findall(b -> circulation(b) == 0, new_blobs))
        push!(len, length(new_blobs))
        push!(tmp, new_state)
    end

    # trim zeros
    @assert all(len .== len[1])
    toremove = intersect(empty_inds...)
    tokeep = filter(i -> i ∉ toremove, 1:len[1])

    @inbounds for (i, state) in enumerate(tmp)
        new_blobs, lesp, tesp = state_to_blobs(state, config.δ)
        tmp[i] = blobs_to_state(new_blobs[tokeep], lesp, tesp)[1:end-1]
    end

	if Ne==1
		X = vcat(X[1:Ny,:], reshape(tmp[1], (size(tmp[1], 1), 1)))
    else
		X = vcat(X[1:Ny,:], hcat(tmp...))
    end
		return X, t + config.Δt
end

function new_vortex(X, t::Float64, Ny, Nx, config::VortexParams, P::Serial)
	Nypx, Ne = size(X)
	@assert Nypx == Ny + Nx "Wrong value of Ny or Nx"
    tmp = Array{Float64,1}[]
    empty_inds = Vector{Int}[]
    len = Int[]
	tesp = config.tesp
	# states = []
	# states  = map(i->vcat(ens.state.S[:,i], tesp),1:Ne)
	# for col in eachcol(ens.state.S)
    @inbounds for i = 1:Ne
		col = view(X, Ny+1:Nypx, i)
        new_state = new_state_equation(vcat(col, tesp), t, config)
		# new_state = state_equation(states[i], t, config)
        new_blobs, lesp, tesp = state_to_blobs(new_state, config.δ)
        push!(empty_inds, findall(b -> circulation(b) == 0, new_blobs))
        push!(len, length(new_blobs))
        push!(tmp, new_state)
    end

    # trim zeros
    @assert all(len .== len[1])
    toremove = intersect(empty_inds...)
    tokeep = filter(i -> i ∉ toremove, 1:len[1])

    @inbounds for (i, state) in enumerate(tmp)
        new_blobs, lesp, tesp = state_to_blobs(state, config.δ)
        tmp[i] = blobs_to_state(new_blobs[tokeep], lesp, tesp)[1:end-1]
    end

	if Ne==1
		X = vcat(X[1:Ny,:], reshape(tmp[1], (size(tmp[1], 1), 1)))
    else
		X = vcat(X[1:Ny,:], hcat(tmp...))
    end
		return X, t + config.Δt
end
