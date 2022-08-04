export lowrank_vortexassim

function lowrank_vortexassim(X)






end




# export lowrank_vortexassim, lowrank_assimilate_obs
#
# function lowrank_vortexassim(X, tspan::Tuple{S,S}, config::VortexParams, idxp, pressure_data; isfiltered::Bool = true, P::Parallel = serial) where {S<:Real}
#
# # Define the additive Inflation
# TransportMap.@get config (ϵX, ϵΓ, ϵLESP, β)
# ϵx = RecipeInflation([ϵX; ϵΓ; ϵLESP])
# ϵmul = MultiplicativeInflation(β)
#
# Ny = size(idxp,1)
# ϵy = AdditiveInflation(zeros(Ny), config.ϵY)
#
# tesp = config.tesp
#
# h(t, x) = measure_state(vcat(x, tesp), t, config, idxp)
# yt(t) = cfd_pressure(t, idxp, config, pressure_data);
#
#
# # Set different times
# Δtobs = config.Δt
# Δtdyn = config.Δt
# t0, tf = tspan
# step = ceil(Int, Δtobs/Δtdyn)
#
# n0 = ceil(Int64, t0/Δtobs) + 1
# J = (tf-t0)/Δtobs
# Acycle = n0:n0+J#-1
#
# # Array dimensions
# Nypx, Ne = size(X)
# Nx = Nypx - Ny
# Ystar = zeros(Ny, Ne)
#
# Xf = Array{Float64,2}[]
# push!(Xf, copy(state(X, Ny, Nx)))
#
# Xa = Array{Float64,2}[]
# push!(Xa, copy(state(X, Ny, Nx)))
#
# # Set few parameters for the transport map
# m = 8
#
#
# # Run particle filter
# @showprogress for i=1:length(Acycle)
# 	@show i
#     # Forecast
# 	@inbounds for j=1:step
# 		tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
# 		_, X = vortex(tj, X, Ny, Nx, config, P)
# 	end
#
# 	# Update state dimension
# 	Nypx = size(X, 1)
# 	Nx = Nypx - Ny
#
# 	push!(Xf, copy(state(X, Ny, Nx)))
#
#     # Get real measurement
#     ystar = yt(t0+i*Δtobs)
#
# 	# Perform additive inflation for each ensemble member
# 	ϵmul(X, Ny+1, Ny+Nx)
# 	if i>5
# 		ϵx(X, Ny, Nx)
# 	end
# 	# Filter state
# 	if isfiltered == true
# 		@inbounds for i=1:Ne
# 			x = view(X, Ny+1:Ny+Nx, :)
# 			x .= filter_state!(x)
# 		end
# 	end
#
# 	if i > 10
# 	# Compute measurements
# 	observe(h, t0+(i-1)*Δtobs, X, Ny, Nx; P = P)
# 	# Perturb the measurements
# 	ϵy(X, 1, Ny)
#
# 	# Construct the informative subspaces in the state and measurement spaces
# 	∇h = 1/config.ϵY*Jacobian(x->h(t0+(i-1)*Δtobs,x), mean(viewstate(X, Ny, Nx); dims = 2)[:,1], Ny, Nx)*
# 	                 Diagonal(std(viewstate(X, Ny, Nx); dims = 2)[:,1])
#
# 	S∇h = svd(∇h)
# 	U = S∇h.U
# 	Σ = S∇h.S
# 	V = Matrix(S∇h.Vt')
#
# 	 # Choose the rank of the approximation based on energy level
# 	energy = 0.9999
# 	r = findfirst(ei -> ei > energy, cumsum(Σ.^2)./sum(Σ.^2))
# 	@show r
# 	# Rescaled the samples
# 	L = LinearTransform(X)
# 	transform!(L, X)
#
# 	# Rescale ystar
# 	ystar .-= view(L.μ,1:Ny)
# 	ystar ./= L.L.diag[1:Ny]
#
# 	# Construct the projections
# 	Xprime = zeros(2*r, Ne)
#
# 	# Project the normalized measurements on Uᵀ
# 	view(Xprime,1:r,:) .=  view(U',1:r,:)*view(X,1:Ny,:)
#
# 	# Project the normalized state on Vᵀ
# 	view(Xprime,r+1:2*r,:) .=  view(V',1:r,:)*view(X,Ny+1:Nx+Ny,:)
#
# 	# Compute the unaffected part of the state x̃⟂ = x̃ - Vx′
# 	for i=1:Ne
# 		view(X,Ny+1:Ny+Nx,i) .-= view(V,:,1:r)*view(Xprime,r+1:2*r,i)
# 	end
#
# 	Ystarprime = repeat(view(U',1:r,:)*ystar, 1, Ne)
#
# 	M = HermiteMap(m, Xprime)
#
# 	optimize(M, Xprime, "kfold"; withqr = true, verbose = true, apply_rescaling = true, start = r + 1, P = P)
#
# 	# Evaluate the transport map
#     F = TransportBasedInference.evaluate(M, Xprime; apply_rescaling = true, start = r+1, P = P)
#
# 	# Generate posterior samples
# 	inverse!(F, M, Xprime, Ystarprime; apply_rescaling = true, start = r+1, P = P)
#
# 	# Add projected posterior to the normalized state
# 	for i=1:Ne
# 		view(X,Ny+1:Ny+Nx,i) .+= view(V,:,1:r)*view(Xprime,r+1:2*r,i)
# 	end
#
# 	itransform!(L, X)
#
# 	# Filter state
# 	if isfiltered == true
# 		@inbounds for i=1:Ne
# 			x = view(X, Ny+1:Ny+Nx, :)
# 			x .= filter_state!(x)
# 		end
# 	end
#
# 	end
#
#     push!(Xa, copy(state(X, Ny, Nx)))
# end
#
# return Xf, Xa
# end
#
# function lowrank_assimilate_obs(M::HermiteMap, X, Ystar, Ny, Nx; withconstant::Bool = false,
#                         withqr::Bool = false, verbose::Bool = false, P::Parallel = serial)
#
#         Nystar, Neystar = size(Ystar)
#         Nypx, Ne = size(X)
#
#         @assert Nystar == Ny "Size of ystar is not consistent with Ny"
#         @assert Nypx == Ny + Nx "Size of X is not consistent with Ny or Nx"
#         @assert Ne == Neystar "Size of X and Ystar are not consistent"
#
#         # Optimize the transport map
#         M = optimize(M, X, 10; withconstant = withconstant, withqr = withqr,
#                                verbose = verbose, start = Ny+1, P = P)
#
#         # Evaluate the transport map
#         F = evaluate(M, X; start = Ny+1, P = P)
#
#         inverse!(F, M, X, Ystar; start = Ny+1, P = P)
#
#         return X
# end
