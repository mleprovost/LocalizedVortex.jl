export dstateobs, dobsobs, localized_vortexassim

# Compute the distance between the different pressure sensors and the position of the different blobs
# The jth column contains:
# 1st line full of zero because the distance between the j-th pressure sensor and itself is zero
# Then the distance is compute by block of 3 lines for the x, y position and circulation of each vortex elements
# The ith line contains the distance  with the different pressure sensors
# This matrix is non-square

#
# function dstateobs(state::Array{Float64,1}, c::ComplexF64, config::VortexParams, idxp)
#     Np = length(idxp)
#     Nx = size(state, 1)
#     plate = Plate(config.N, config.L, c, config.α);
#     zp = plate.zs[idxp]
#     # last line is the impact of pressure sensors on the LESPc
#     zLE = c + 0.5config.L*exp(im*config.α)
#
#     Nb = ceil(Int64,(length(state)-1)/3)
#     zb = map(i->state[3i-2] + im*state[3i-1], 1:Nb)
#
#     d = zeros(Nx, Np)
#     # dtmp = zeros(Nx)
#     for j=1:Np
#         # dtmp
#         for i=1:Nb
#             d[3*i-2:3*i,j] .= abs(zp[j]-zb[i])
#         end
#         d[end,j] = abs(zLE-zp[j])
#         # d[:,j] = sortperm(dtmp)
#     end
#
#     return d
# end


function dstateobs(X, Ny, Nx, c::ComplexF64, config::VortexParams, idxp)
    Nypx, Ne = size(X)
    @assert Nypx == Ny + Nx
    @assert Ny == length(idxp)
	Nv =  ceil(Int64,(Nx-1)/3)

    dXY = zeros(Nx, Ny, Ne)

	plate = Plate(config.N, config.L, c, config.α);
	zp = plate.zs[idxp]
	# last line is the impact of pressure sensors on the LESPc
	zLE = c + 0.5config.L*exp(im*config.α)

    for i=1:Ne
        xi = X[Ny+1:Ny+Nx, i]
        zi = map(l->xi[3*l-2] + im*xi[3*l-1], 1:Nv)

		for k=1:Ny
			# Compute distance between point vortices and pressure sensors
        	for J=1:Nv
                dXY[3*J-2:3*J,k,i] .= abs(zi[J] - zp[k])
            end

			# Compute distance between LESP (located at the leading edge) and the pressure sensors
			dXY[end,k,i] = abs(zLE - zp[k])
        end

    end
    return mean(dXY, dims = 3)[:,:,1]
end

# Compute the distance between the different pressure sensors
function dobsobs(c::ComplexF64, config::VortexParams, idxp)

    Np = length(idxp)
    plate = Plate(config.N, config.L, c, config.α);
    zp = plate.zs[idxp]

    d = zeros(Np,Np)

    for j=1:Np
        for i=1:j
            dij = abs(zp[i] - zp[j])
            d[i,j] = dij
            d[j,i] = dij
        end
    end

    return d
end


# Create a function to perform the sequential assimilation for any sequential filter SeqFilter
function localized_vortexassim(algo::StochEnKF, Lxy, Lyy, X, tspan::Tuple{S,S}, config::VortexParams, idxp, pressure_data; isfiltered::Bool = true, P::Parallel = serial, nbaby::Int64 = 0) where {S<:Real}

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

	# Compute the distance matrix for the sensors
	dyy = dobsobs(complex(0.0), config, idxp)
	Gyy = gaspari.(dyy./Lyy)

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

	   # Generate samples from the observation noise
	   ϵ = algo.ϵy.σ*randn(Ny, Ne) .+ algo.ϵy.m

	   # Form the perturbation matrix for the state
	   Xpert = (1/sqrt(Ne-1))*(X[Ny+1:Ny+Nx,:] .- mean(X[Ny+1:Ny+Nx,:]; dims = 2)[:,1])
	   # Form the perturbation matrix for the observation
	   HXpert = (1/sqrt(Ne-1))*(X[1:Ny,:] .- mean(X[1:Ny,:]; dims = 2)[:,1])
	   # Form the perturbation matrix for the observation noise
	   ϵpert = (1/sqrt(Ne-1))*(ϵ .- mean(ϵ; dims = 2)[:,1])
	   # Kenkf = Xpert*HXpert'*inv(HXpert*HXpert'+ϵpert*ϵpert')

	   # Apply the Kalman gain based on the representers
	   # Burgers G, Jan van Leeuwen P, Evensen G. 1998 Analysis scheme in the ensemble Kalman
	   # filter. Monthly weather review 126, 1719–1724. Solve the linear system for b ∈ R^{Ny × Ne}:

	   Σy = (HXpert*HXpert' + ϵpert*ϵpert')
	   localizedΣy =  Gyy .* Σy

	   b = (HXpert*HXpert' + ϵpert*ϵpert')\(ystar .- (X[1:Ny,:] + ϵ))

	   # Update the ensemble members according to:
	   # x^{a,i} = x^i - Σ_{X,Y}b^i, with b^i =  Σ_Y^{-1}(h(x^i) + ϵ^i - ystar)
	   dxy = dstateobs(X, Ny, Nx, complex(t0+i*Δtobs), config, idxp)
	   Gxy = gaspari.(dxy./Lxy)
	   localizedΣxy = (Xpert*HXpert')

	   # Update the ensemble members according to:
	   # x^{a,i} = x^i - Σ_{X,Y}b^i, with b^i =  Σ_Y^{-1}(h(x^i) + ϵ^i - ystar)
	   view(X,Ny+1:Ny+Nx,:) .+= localizedΣxy*b

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
