export lowrankvortexassim

function lowrankvortexassim(algo::SeqFilter, X, tspan::Tuple{S,S}, config::VortexParams, idxp, pressure_data; isfiltered::Bool = true, P::Parallel = serial, nbaby::Int64 = 0) where {S<:Real}

# Define the additive Inflation
ϵX = config.ϵX
ϵΓ = config.ϵΓ
ϵLESP = config.ϵLESP
β = config.β


Nesub = 200

Ny = size(idxp,1)

ϵx = RecipeInflation([ϵX; ϵΓ; ϵLESP])
ϵmul = MultiplicativeInflation(β)
# ϵy = AdditiveInflation(zeros(Ny), config.ϵY)
tesp = config.tesp
h(x, t) = measure_state(vcat(x, tesp), t, config, idxp)
hcoarse(x, t) = measure_statecoarse(vcat(x, tesp), t, config, idxp)

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
for i=1:length(Acycle)
	@show i
	# Forecast
	@inbounds for j=1:step
	   tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
	   X, _ = vortex2(X, tj, Ny, Nx, config, P)
	end

	# Update state dimension
	Nypx = size(X, 1)
	Nx = Nypx - Ny

	push!(Xf, copy(state(X, Ny, Nx)))

	# Perform additive inflation for each ensemble member
	ϵmul(X, Ny+1, Ny+Nx)
	if i>5 || t0 != 0.0
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
	observe(h, X, t0+i*Δtobs, Ny, Nx; P = P)

	# Perturb measurements
	# ϵy(X, 1, Ny)

	Cx = zeros(Nx, Nx)
	Cy  = zeros(Ny, Ny)
	Jac = zeros(Ny, Nx)

	Dx = Diagonal(std(view(X, Ny+1:Ny+Nx, :); dims = 2)[:,1])
	# Dy = Diagonal(std(view(X, 1:Ny, :); dims = 2)[:,1])

	# Dx = cholesky(cov(state(X, Ny, Nx)')).L

	subset = rand(1:Ne, Nesub)
	cache = CacheJacobian(Ny, Nx; fdtype = :forward)

	for j in subset
		Jacobian!(Jac, x-> hcoarse(x, t0+(i-1)*Δtobs), X[Ny+1:Ny + Nx,j], cache; fdtype = :forward)
	    Cx .+= (1/Nesub)*(1/config.ϵY*Jac*Dx)'*(1/config.ϵY*Jac*Dx)
	    Cy .+= (1/Nesub)*(1/config.ϵY*Jac*Dx)*(1/config.ϵY*Jac*Dx)'
	end

	ry = min(6, Ny)
	rx = 6

	Lbx, V = pheig(Symmetric(Cx), rank = rx)
	V = reverse(V, dims = 2)

	Lby, U = pheig(Symmetric(Cy), rank = ry)
	U = reverse(U, dims = 2)

	@show Lbx, Lby

	# Rescaled the samples
	Xscaled = copy(X)
	L = LinearTransform(Xscaled; diag = true)
	transform!(L, Xscaled)

	# Get real measurement
	ystar .= yt(t0+i*Δtobs)

	# Rescale ystar
	ystar .-= view(L.μ,1:Ny)
	ystar ./= L.L.diag[1:Ny]

	ystarprime = U'*ystar

	# Construct the state and measurement projections
	Xprime = zeros(ry + rx, Ne)

	# Project the normalized measurements on Uᵀ
	view(Xprime,1:ry,:) .=  deepcopy(U'*Xscaled[1:Ny,:])

	# Project the normalized state on Vᵀ
	view(Xprime,ry+1:ry + rx,:) .=  deepcopy(V'*Xscaled[Ny+1:Ny+Nx,:])

	# Compute the unaffected part of the state x̃⟂ = x̃ - Vx′
	Xperp = copy(Xscaled)
	for i=1:Ne
		view(Xperp, Ny+1:Ny+Nx,i) .-= V*deepcopy(Xprime[ry+1:ry + rx,i])
	end

	# Perform the assimilation in the rescaled space
	# Xprime = algo(Xprime, ystarprime, t0+i*Δtobs)


	# Add projected posterior to the normalized state
	Xpost = zero(Xscaled)

	for i=1:Ne
		view(Xpost,Ny+1:Ny+Nx,i) .=  V*view(Xprime,ry+1:ry + rx,i) + view(Xperp,Ny+1:Ny+Nx,i)
	end

	# Unnormalized the data
	itransform!(L, Xscaled)
	itransform!(L, Xpost)

	X = copy(Xpost)

	# Filter state
	if isfiltered == true
	   @inbounds for i=1:Ne
		   x = view(X, Ny+1:Ny+Nx, i)
		   x .= filter_state!(x)
	   end
	end

	push!(Xa, copy(state(X, Ny, Nx)))
	end

	return Xf, Xa
end
