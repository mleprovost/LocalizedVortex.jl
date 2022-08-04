import TransportBasedInference: state, viewstate, meas, viewmeas, Parallel, Serial, Thread

# Write tools for data assimilation of the vortex model
export filter_state!, RecipeInflation, vortexassim, vortexassim2, vortexmean_assim, nlvortexassim



# Filtering function to bound the LESPC function
function filter_state!(x)#; lowerbound = 0.0, upperbound = 3.0)
    x[end] = clamp(x[end], 0.0, 3.0)#lowerbound, upperbound)
    return x
end

# Custom Inflation

struct RecipeInflation <: InflationType
    "Parameters"
    p::Array{Float64,1}
end

# This function apply additive inflation to the state components only,
# not the measurements, X is an Array{Float64,2} or a view of it
function (ϵ::RecipeInflation)(X, Ny, Nx; laplace::Bool=false)
	ϵX, ϵΓ, ϵLESP = ϵ.p
	if laplace == false
		for col in eachcol(X)
			for i in 1:3:Nx-2
			col[Ny + i: Ny + i + 1] .+= ϵX*randn(2)
			col[Ny + i + 2]   += ϵΓ*randn()
			end
			col[Ny + Nx] += ϵLESP*randn()
		end
	else
		for col in eachcol(X)
			for i in 1:3:Nx-2
			col[Ny + i: Ny + i + 1] .+= sqrt(2.0)*ϵX*rand(Laplace(), 2)
			col[Ny + i + 2]   += sqrt(2.0)*ϵΓ*rand(Laplace())
			end
			col[Ny + Nx] += sqrt(2.0)*ϵLESP*rand(Laplace())
		end
	end
end

# Create a function to perform the sequential assimilation for any sequential filter SeqFilter
function vortexassim2(algo::SeqFilter, X, tspan::Tuple{S,S}, config::VortexParams, idxp, pressure_data; isfiltered::Bool = true, P::Parallel = serial, nbaby::Int64 = 0) where {S<:Real}

# Define the additive Inflation
ϵX = config.ϵX
ϵΓ = config.ϵΓ
ϵLESP = config.ϵLESP
β = config.β

Ny = size(idxp,1)

ϵx = RecipeInflation([ϵX; ϵΓ; ϵLESP])
ϵmul = MultiplicativeInflation(β)
# ϵy = AdditiveInflation(zeros(Ny), config.ϵY)
tesp = config.tesp
h(x, t) = measure_state(vcat(x, tesp), t, config, idxp)
yt(t) = cfd_pressure(t, idxp, config, pressure_data);

# Set different times
Δtobs = algo.Δtobs
Δtdyn = algo.Δtdyn
t0, tf = tspan
step = ceil(Int, Δtobs/Δtdyn)

n0 = ceil(Int64, t0/Δtobs) + 1
J = (tf-t0)/Δtobs
Acycle = n0:n0+J#-1

# Array dimensions
Nypx, Ne = size(X)
Nx = Nypx - Ny
ystar = zeros(Ny)

Xf = Array{Float64,2}[]
push!(Xf, copy(state(X, Ny, Nx)))

Xa = Array{Float64,2}[]
push!(Xa, copy(state(X, Ny, Nx)))

# Run particle filter
@showprogress for i=1:length(Acycle)
   # Forecast
   @inbounds for j=1:step
	   tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
	   X, _ = vortex2(X, tj, Ny, Nx, config, P)
   end

   # Update state dimension
   Nypx = size(X, 1)
   Nx = Nypx - Ny

   push!(Xf, deepcopy(state(X, Ny, Nx)))

   # Get real measurement
   ystar .= yt(t0+i*Δtobs)

   # Perform additive inflation for each ensemble member
   ϵmul(X, Ny+1, Ny+Nx)
   if i>5
	   ϵx(X, Ny, Nx)
   end
   # Filter state
   if isfiltered == true
	   @inbounds for i=1:Ne
		   x = view(X, Ny+1:Ny+Nx, i)
		   x .= filter_state!(x)
	   end
   end
   # @show t0+i*Δtobs
   # Compute measurements
   # observe(h, X, t0+(i-1)*Δtobs, Ny, Nx; P = P)
   observe(h, X, t0+i*Δtobs, Ny, Nx; P = P)

   # if isfiltered == true
	#    outliers = iqrfilter(X[1:Ny, :])
	#    @show length(outliers)
	#    replaceoutliers!(X, outliers)
   # end

   # Remove the last ten shed vortices from the inference
   # The last line contains the lespc estimate
   # nblob = (Nx-1)÷3
   # if nblob > nbaby
	#    # Measurement + old vortices + lesp estimate
	#    Xinference = vcat(X[1:Ny+3*(nblob-nbaby),:], X[end:end,:])
   # else
	#    # Measurements + lesp estimate
	#    Xinference = vcat(X[1:Ny,:], X[end:end,:])
   # end
	# @show size(Xinference,1)-Nypx
   # Generate posterior samples
   X = algo(X, ystar, t0+i*Δtobs)
	# Xinference = algo(Xinference, ystar, t0+i*Δtobs)

   # if nblob > nbaby
	#    # Update the old vortices
	#    X[Ny+1:Ny+3*(nblob-nbaby),:] .=  copy(Xinference[Ny+1:Ny+3*(nblob-nbaby),:])
	#    # Update the lesp value
	#    X[end,:] .= copy(Xinference[end,:])
   # else
	#    # Update only the lesp value
	#    X[end,:] .= copy(Xinference[end,:])
   # end
   # Filter state
   if isfiltered == true
	   @inbounds for i=1:Ne
		   x = view(X, Ny+1:Ny+Nx, i)
		   x .= filter_state!(x)
	   end
   end

   push!(Xa, deepcopy(state(X, Ny, Nx)))
end

return Xf, Xa
end

# Create a function to perform the sequential assimilation for any sequential filter SeqFilter
function vortexassim(algo::SeqFilter, X, tspan::Tuple{S,S}, config::VortexParams, idxp, pressure_data; isfiltered::Bool = true, P::Parallel = serial) where {S<:Real}

# Define the additive Inflation
ϵX = config.ϵX
ϵΓ = config.ϵΓ
ϵLESP = config.ϵLESP
β = config.β

Ny = size(idxp,1)

ϵx = RecipeInflation([ϵX; ϵΓ; ϵLESP])
ϵmul = MultiplicativeInflation(β)
# ϵy = AdditiveInflation(zeros(Ny), config.ϵY)
tesp = config.tesp
h(x, t) = measure_state(vcat(x, tesp), t, config, idxp)
yt(t) = cfd_pressure(t, idxp, config, pressure_data);

# Set different times
Δtobs = algo.Δtobs
Δtdyn = algo.Δtdyn
t0, tf = tspan
step = ceil(Int, Δtobs/Δtdyn)

n0 = ceil(Int64, t0/Δtobs) + 1
J = (tf-t0)/Δtobs
Acycle = n0:n0+J#-1

# Array dimensions
Nypx, Ne = size(X)
Nx = Nypx - Ny
ystar = zeros(Ny)

Xf = Array{Float64,2}[]
push!(Xf, copy(state(X, Ny, Nx)))

Xa = Array{Float64,2}[]
push!(Xa, copy(state(X, Ny, Nx)))

# Run particle filter
@showprogress for i=1:length(Acycle)
   # Forecast
   @inbounds for j=1:step
	   tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
	   X, _ = vortex(X, tj, Ny, Nx, config, P)
   end

   # Update state dimension
   Nypx = size(X, 1)
   Nx = Nypx - Ny

   push!(Xf, deepcopy(state(X, Ny, Nx)))

   # Get real measurement
   ystar .= yt(t0+i*Δtobs)

   # Perform additive inflation for each ensemble member
   ϵmul(X, Ny+1, Ny+Nx)
   if i>5
	   ϵx(X, Ny, Nx)
   end
   # Filter state
   if isfiltered == true
	   @inbounds for i=1:Ne
		   x = view(X, Ny+1:Ny+Nx, i)
		   x .= filter_state!(x)
	   end
   end

   # Compute measurements
   # observe(h, X, t0+(i-1)*Δtobs, Ny, Nx; P = P)
   observe(h, X, t0+i*Δtobs, Ny, Nx; P = P)

   # Generate posterior samples
   X = algo(X, ystar, t0+i*Δtobs)

   # Filter state
   if isfiltered == true
	   @inbounds for i=1:Ne
		   x = view(X, Ny+1:Ny+Nx, i)
		   x .= filter_state!(x)
	   end
   end

   push!(Xa, deepcopy(state(X, Ny, Nx)))
end

return Xf, Xa
end


# Create a function to perform the sequential assimilation for any sequential filter SeqFilter
function vortexmean_assim(algo::SeqFilter, X, tspan::Tuple{S,S}, config::VortexParams, idxp, pressure_data; isfiltered::Bool = true, P::Parallel = serial) where {S<:Real}

# Define the additive Inflation
ϵX = config.ϵX
ϵΓ = config.ϵΓ
ϵLESP = config.ϵLESP
β = config.β

Ny = size(idxp,1)

ϵx = RecipeInflation([ϵX; ϵΓ; ϵLESP])
ϵmul = MultiplicativeInflation(β)
# ϵy = AdditiveInflation(zeros(Ny), config.ϵY)
tesp = config.tesp
h(t, x) = measure_state(vcat(x, tesp), t, config, idxp)
yt(t) = cfd_pressure(t, idxp, config, pressure_data);

# Set different times
Δtobs = algo.Δtobs
Δtdyn = algo.Δtdyn
t0, tf = tspan
step = ceil(Int, Δtobs/Δtdyn)

n0 = ceil(Int64, t0/Δtobs) + 1
J = (tf-t0)/Δtobs
Acycle = n0:n0+J#-1

# Array dimensions
Nypx, Ne = size(X)
Nx = Nypx - Ny
ystar = zeros(Ny)

Xf = Array{Float64,2}[]
push!(Xf, copy(state(X, Ny, Nx)))

Xa = Array{Float64,2}[]
push!(Xa, copy(state(X, Ny, Nx)))

# Run particle filter
@showprogress for i=1:length(Acycle)
   # Forecast
   @inbounds for j=1:step
	   tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
	   X, _ = vortex_mean(X, tj, Ny, Nx, config, P)
   end

   # Update state dimension
   Nypx = size(X, 1)
   Nx = Nypx - Ny

   push!(Xf, deepcopy(state(X, Ny, Nx)))

   # Get real measurement
   ystar .= yt(t0+i*Δtobs)

   # Perform additive inflation for each ensemble member
   ϵmul(X, Ny+1, Ny+Nx)
   if i>5
	   ϵx(X, Ny, Nx)
   end
   # Filter state
   if isfiltered == true
	   @inbounds for i=1:Ne
		   x = view(X, Ny+1:Ny+Nx, i)
		   x .= filter_state!(x)
	   end
   end

   # Compute measurements
   # observe(h, X, t0+(i-1)*Δtobs, Ny, Nx; P = P)
   observe(h, t0+i*Δtobs, X, Ny, Nx; P = P)

   # Generate posterior samples
   X = algo(X, ystar, t0+i*Δtobs)

   # Filter state
   if isfiltered == true
	   @inbounds for i=1:Ne
		   x = view(X, Ny+1:Ny+Nx, i)
		   x .= filter_state!(x)
	   end
   end

   push!(Xa, deepcopy(state(X, Ny, Nx)))
end

return Xf, Xa
end


# Create a function to perform the sequential assimilation for any sequential filter SeqFilter
function nlvortexassim(X, tspan::Tuple{S,S}, config::VortexParams, idxp, pressure_data; isfiltered::Bool = true, P::Parallel = serial) where {S<:Real}

# Define the additive Inflation
ϵX = config.ϵX
ϵΓ = config.ϵΓ
ϵLESP = config.ϵLESP
β = config.β

ϵx = RecipeInflation([ϵX; ϵΓ; ϵLESP])
ϵmul = MultiplicativeInflation(β)

Ny = size(idxp,1)
ϵy = AdditiveInflation(zeros(Ny), config.ϵY)

tesp = config.tesp

h(t, x) = measure_state(vcat(x, tesp), t, config, idxp)
yt(t) = cfd_pressure(t, idxp, config, pressure_data);


# Set different times
Δtobs = config.Δt
Δtdyn = config.Δt
t0, tf = tspan
step = ceil(Int, Δtobs/Δtdyn)

n0 = ceil(Int64, t0/Δtobs) + 1
J = (tf-t0)/Δtobs
Acycle = n0:n0+J#-1

# Array dimensions
Nypx, Ne = size(X)
Nx = Nypx - Ny
Ystar = zeros(Ny, Ne)

Xf = Array{Float64,2}[]
push!(Xf, copy(state(X, Ny, Nx)))

Xa = Array{Float64,2}[]
push!(Xa, copy(state(X, Ny, Nx)))

# Set few parameters for the transport map
m = 30


# Run particle filter
@showprogress for i=1:length(Acycle)
	@show i
    # Forecast
	@inbounds for j=1:step
		tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
		X, _ = vortex(X, tj, Ny, Nx, config, P)
	end

	# Update state dimension
	Nypx = size(X, 1)
	Nx = Nypx - Ny

	push!(Xf, copy(state(X, Ny, Nx)))

    # Get real measurement
    Ystar .= repeat(yt(t0+i*Δtobs), 1, Ne)

	# Perform additive inflation for each ensemble member
	ϵmul(X, Ny+1, Ny+Nx)
	if i>5
		ϵx(X, Ny, Nx)
	end
	# Filter state
	if isfiltered == true
		@inbounds for i=1:Ne
			x = view(X, Ny+1:Ny+Nx, :)
			x .= filter_state!(x)
		end
	end

	# Compute measurements
	observe(h, X, t0+(i-1)*Δtobs, Ny, Nx; P = P)

	# Perturb the measurements
	ϵy(X, 1, Ny)
    # Generate posterior samples
	M = HermiteMap(m, X)
	X = assimilate_obs(M, X, Ystar, Ny, Nx; withqr = true)

	# Filter state
	if isfiltered == true
		@inbounds for i=1:Ne
			x = view(X, Ny+1:Ny+Nx, :)
			x .= filter_state!(x)
		end
	end

    push!(Xa, copy(state(X, Ny, Nx)))
end

return Xf, Xa
end
