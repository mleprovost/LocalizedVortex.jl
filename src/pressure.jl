# This file will contains the routine to compute pressure
# and pressure jump across an infinitely thin plate

using PotentialFlow: Points, Plates


export cfd_pressure, measure_state, measure_statecoarse, measure_state_bernoulli, Jl_l

function cfd_pressure(t, taps, config, pressure_data)
    s = [-0.5config.L*cos(n*π/(config.N-1)) for n in taps]
    [-pressure_data(i,t) for i in s]
end

### Tools compute pressure

function measure_state(state, t, config, taps = 1:config.N)
    blobs, lesp, tesp = state_to_blobs(state, config.δ)
    # plate = Plate(config.N, config.L, complex(t+config.Δt), config.α)
    plate = Plate(config.N, config.L, complex(t), config.α)

    motion = Plates.RigidBodyMotion(config.ċ, 0.0)

    z₊ = (blobs[end-1].z + 2plate.zs[end])/3
    z₋ = (blobs[end].z + 2plate.zs[1])/3
    ss = Plates.Chebyshev.nodes(config.N)[taps]
    Plates.surface_pressure_inst(plate, motion, blobs, (z₊, z₋), t, config.Δt, lesp, tesp, ss)
end

function measure_statecoarse(state, t, config, taps = 1:config.N)
    blobs, lesp, tesp = state_to_blobs(state, config.δ)
    # plate = Plate(128, config.L, complex(t+config.Δt), config.α)
    plate = Plate(128, config.L, complex(t), config.α)

    motion = Plates.RigidBodyMotion(config.ċ, 0.0)

    z₊ = (blobs[end-1].z + 2plate.zs[end])/3
    z₋ = (blobs[end].z + 2plate.zs[1])/3
    ss = Plates.Chebyshev.nodes(config.N)[taps]
    Plates.surface_pressure_inst(plate, motion, blobs, (z₊, z₋), t, config.Δt, lesp, tesp, ss)
end

# function measure_state(state, t, config, taps = 1:config.N)
#     blobs, lesp, tesp = state_to_blobs(state, config.δ)
#     plate = Plate(config.N, config.L, complex(t + config.Δt), config.α)
#     motion = Plates.RigidBodyMotion(config.ċ, 0.0)
#
#     z₊ = (blobs[end-1].z + 2plate.zs[end])/3
#     z₋ = (blobs[end].z + 2plate.zs[1])/3
#
#     Plates.surface_pressure_inst(plate, motion, blobs, (z₊, z₋), t, config.Δt, lesp, tesp)[taps]
# end

function measure_state_bernoulli(state, t, config, Jtab, taps)
    blobs, lesp, tesp = state_to_blobs(state, config.δ)
    plate = Plate(config.N, config.L, complex(config.ċ*(t + config.Δt)), config.α)
    motion = Plates.RigidBodyMotion(config.ċ, 0.0)

    # if isempty(blobs)
    #     Δz₊ = Δz₋ = config.Δt
    #     z₊ = plate.zs[end] + im*Δz₊*exp(im*plate.α)
    #     z₋ = plate.zs[1] - Δz₋*exp(im*plate.α)
    # else
        z₊ = (blobs[end-1].z + 2plate.zs[end])/3
        z₋ = (blobs[end].z + 2plate.zs[1])/3
    # end
    w₊, w₋, ϕ̇₊, ϕ̇₋ = ddtcomplexpotential(plate, motion, blobs, (z₊, z₋), t, config.Δt, lesp, tesp, Jtab)

    # puv₊ = 0.5*(abs(motion.ċ)^2 .- abs.(w₊[taps] .-motion.ċ).^2)
    # plv₊ = 0.5*(abs(motion.ċ)^2 .- abs.(w₋[taps] .-motion.ċ).^2)


    Δpbv₊ = -0.5*(abs.(w₊[taps] .-motion.ċ).^2 - abs.(w₋[taps] .-motion.ċ).^2)
    ΔpbΓ₊ = -(ϕ̇₊[taps] - ϕ̇₋[taps])
    Δpb₊ = Δpbv₊ + ΔpbΓ₊

    return Δpb₊ #Δpbv₊, ΔpbΓ₊, Δpb₊
    #Plates.surface_pressure_inst(plate, motion, blobs, (z₊, z₋), t, config.Δt, lesp, tesp)[taps]
end

function measure_state_bernoulli(state, t, config, Jtab)
    blobs, lesp, tesp = state_to_blobs(state, config.δ)
    plate = Plate(config.N, config.L, complex(config.ċ*(t + config.Δt)), config.α)
    motion = Plates.RigidBodyMotion(config.ċ, 0.0)

    z₊ = (blobs[end-1].z + 2plate.zs[end])/3
    z₋ = (blobs[end].z + 2plate.zs[1])/3

    w₊, w₋, ϕ̇₊, ϕ̇₋ = ddtcomplexpotential(plate, motion, blobs, (z₊, z₋), t, config.Δt, lesp, tesp, Jtab)

    Δpbv₊ = -0.5*(abs.(w₊ .-motion.ċ).^2 - abs.(w₋ .-motion.ċ).^2)
    ΔpbΓ₊ = -(ϕ̇₊ - ϕ̇₋)
    Δpb₊ = Δpbv₊ + ΔpbΓ₊

    return Δpb₊ #Δpbv₊, ΔpbΓ₊, Δpb₊
end

function induce_velocityonplate(p::Plate, t)
    # From Darwin's thesis appendix
    @get p (N, α, L, c, B₀, B₁, Γ, A, C, ss)

    w = Plates.Chebyshev.firstkind(real.(C), ss) * tangent(p)
    BLAS.axpy!(1, (B₀ .+ B₁ * ss) * normal(p), w)

    wγ = 0.5*Plates.strength(p)*tangent(p)

    # w₊ = w - wγ, w₋ = w + wγ
    # note that this provides u+iv (JDE) in the inertial reference frame
    return w - wγ, w + wγ
end

function induce_velocityonplate(p::Plate, t, ss::AbstractArray{T}) where {T <: Real}
    # From Darwin's thesis appendix
    @get p (N, α, L, c, B₀, B₁, Γ, A, C)

    w = Plates.Chebyshev.firstkind(real.(C), ss) * tangent(p)
    BLAS.axpy!(1, (B₀ .+ B₁ * ss) * normal(p), w)

    wγ = 0.5*Plates.strength(p, ss)*tangent(p)

    # w₊ = w - wγ, w₋ = w + wγ
    # note that this provides u+iv (JDE) in the inertial reference frame
    return w - wγ, w + wγ
end

function ddtcomplexpotential(p::Plate, ṗ, ambient_sys, z_new, t, Δt, lesp, tesp, Jtab::Array{ComplexF64,2})
        @get p (N, L, C, α, dchebt!)
        @get ṗ (ċ, α̇ , c̈)
        # δ = 5e-3
        # Get Ċ from movement of existing vortex blobs (without vortex shedding)
        Plates.enforce_no_flow_through!(p, ṗ, ambient_sys, t)

        srcvel = self_induce_velocity(ambient_sys, t)
        induce_velocity!(srcvel, ambient_sys, p, t)

        targvel = fill(ċ, length(p))
        Ċ = zero(targvel)

        Plates.induce_acc!(Ċ, p.zs, targvel, ambient_sys, srcvel)

        z₊, z₋ = z_new
        point₊ = Vortex.Point(z₊, 1.0)
        point₋ = Vortex.Point(z₋, 1.0)
        # point₊ = Vortex.Blob(z₊, 1.0, δ)
        # point₋ = Vortex.Blob(z₋, 1.0, δ)

        Γ₊, Γ₋, ∂C₊, ∂C₋ = Plates.vorticity_flux!(p, point₊, point₋, t, lesp, tesp)

        point₊ = Vortex.Point(z₊, Γ₊)
        point₋ = Vortex.Point(z₋, Γ₋)
        # point₊ = Vortex.Blob(z₊, Γ₊, δ)
        # point₋ = Vortex.Blob(z₋, Γ₋, δ)

        w₊, w₋ = induce_velocityonplate(p, t)
        # w₊ = induce_velocity(p, (p, ambient_sys, blob₊, blob₋), t) -0.5*Plates.strength(plate)*tangent(plate)
        # w₋ = induce_velocity(p, (p, ambient_sys, blob₊, blob₋), t) +0.5*Plates.strength(plate)*tangent(plate)

        n̂ = exp(-im*α)
        rmul!(Ċ, n̂)
        dchebt! * Ċ

        @. Ċ += (∂C₊ + ∂C₋)/Δt - im*α̇ *C

        Ȧ = MappedVector(imag, Ċ, 1)

        # Plate is moving at a varying velocity
        τ = exp(im*α)
        n = im*exp(im*α)
        Ḃ₀ = - α̇  * real(conj(τ)*ċ) + real(conj(n)*c̈)
        # Plate is moving at a fixed angle
        Ḃ₁ = 0.0

        # From p.444
        # J₊ = [s - im*√(1-s^2) for s in p.ss]
        # J₋ = [s + im*√(1-s^2) for s in p.ss]

        ϕ̇₊ = real(ϕtabdot(N, L, Ȧ, Ḃ₀, Ḃ₁, -(Γ₊+Γ₋)/Δt, Jtab))
        ϕ̇p = deepcopy(ϕ̇₊)
        ϕ̇₋  = -deepcopy(ϕ̇₊)
        ϕ̇₊ .+= Γ₊/Δt

        # Remove this part since we are accounting twice for the same quantity
        # Add dϕ̇/dt due to the presence of ambient vorticity
        # # zss = p.c .+ p.L/2*ss*exp(im*p.α)
        # # # Γ̇v = 1/(2*π*im)*(Γ₊*log.(zss .- z₊) .+  Γ₋*log.(zss .- z₋))/Δt
        z̃₊ = (z₊ - p.c)*exp(-im*p.α)
        z̃₋ = (z₋ - p.c)*exp(-im*p.α)
        Γ̇v = 1/(2*π*im)*(Γ₊*log.((p.L/2)*p.ss .- z̃₊) .+  Γ₋*log.((p.L/2)*p.ss .- z̃₋))/Δt
        #
        ϕ̇₊ .+= real(Γ̇v)
        ϕ̇₋ .+= real(Γ̇v)
        # # # #
        # # # # # add (1/2*\pi*im)*ΓJ*(z\dot -zJ\dot)/(z - zJ)
        żv = zeros(ComplexF64, length(p.ss))
        # # # # #
        if isempty(ambient_sys) == false
        for (i, source) in enumerate(ambient_sys)
            # żv .+= source.S*Points.cauchy_kernel.(conj.(zss .- source.z)) .*
            #         (ċ .- srcvel[i])
            # żv .+= source.S*conj.(Vortex.Blobs.blob_kernel.(p.L/2*ss .- (source.z-p.c)*exp(-im*p.α), 5e-3)) .*
            #         (ċ .- srcvel[i])*exp(im*p.α)
            żv .+= source.S*conj.(Points.cauchy_kernel.(p.L/2*p.ss .- (source.z-p.c)*exp(-im*p.α))) .*
                    (ċ .- srcvel[i])*exp(-im*p.α)
        end
        end

        ϕ̇₊ .+= real(żv)
        ϕ̇₋ .+= real(żv)

        return w₊, w₋, ϕ̇₊, ϕ̇₋
end

function ddtcomplexpotential(p::Plate, ṗ, ambient_sys, z_new, t, Δt, lesp, tesp,
                             Jtab::Array{ComplexF64,2}, ss::AbstractArray{T}) where {T <: Real}
         @get p (N, L, C, α, dchebt!)
         @get ṗ (ċ, α̇ , c̈)
         # δ = 5e-3
         # Get Ċ from movement of existing vortex blobs (without vortex shedding)
         Plates.enforce_no_flow_through!(p, ṗ, ambient_sys, t)

         srcvel = self_induce_velocity(ambient_sys, t)
         induce_velocity!(srcvel, ambient_sys, p, t)

         targvel = fill(ċ, length(p))
         Ċ = zero(targvel)

         Plates.induce_acc!(Ċ, p.zs, targvel, ambient_sys, srcvel)

         z₊, z₋ = z_new
         point₊ = Vortex.Point(z₊, 1.0)
         point₋ = Vortex.Point(z₋, 1.0)
         # point₊ = Vortex.Blob(z₊, 1.0, δ)
         # point₋ = Vortex.Blob(z₋, 1.0, δ)

         Γ₊, Γ₋, ∂C₊, ∂C₋ = Plates.vorticity_flux!(p, point₊, point₋, t, lesp, tesp)

         point₊ = Vortex.Point(z₊, Γ₊)
         point₋ = Vortex.Point(z₋, Γ₋)
         # point₊ = Vortex.Blob(z₊, Γ₊, δ)
         # point₋ = Vortex.Blob(z₋, Γ₋, δ)

         w₊, w₋ = induce_velocityonplate(p, t, ss)
         # w₊ = induce_velocity(p, (p, ambient_sys, blob₊, blob₋), t) -0.5*Plates.strength(plate)*tangent(plate)
         # w₋ = induce_velocity(p, (p, ambient_sys, blob₊, blob₋), t) +0.5*Plates.strength(plate)*tangent(plate)

         n̂ = exp(-im*α)
         rmul!(Ċ, n̂)
         dchebt! * Ċ

         @. Ċ += (∂C₊ + ∂C₋)/Δt - im*α̇ *C

         Ȧ = MappedVector(imag, Ċ, 1)

         # Plate is moving at a varying velocity
         τ = exp(im*α)
         n = im*exp(im*α)
         Ḃ₀ = - α̇  * real(conj(τ)*ċ) + real(conj(n)*c̈)
         # Plate is moving at a fixed angle
         Ḃ₁ = 0.0

         # From p.444
         # J₊ = [s - im*√(1-s^2) for s in p.ss]
         # J₋ = [s + im*√(1-s^2) for s in p.ss]

         ϕ̇₊ = real(ϕtabdot(N, L, Ȧ, Ḃ₀, Ḃ₁, -(Γ₊+Γ₋)/Δt, Jtab))
         ϕ̇p = deepcopy(ϕ̇₊)
         ϕ̇₋  = -deepcopy(ϕ̇₊)
         ϕ̇₊ .+= Γ₊/Δt

         # Remove this part since we are accounting twice for the same quantity
         # Add dϕ̇/dt due to the presence of ambient vorticity
         # # zss = p.c .+ p.L/2*ss*exp(im*p.α)
         # # # Γ̇v = 1/(2*π*im)*(Γ₊*log.(zss .- z₊) .+  Γ₋*log.(zss .- z₋))/Δt
         z̃₊ = (z₊ - p.c)*exp(-im*p.α)
         z̃₋ = (z₋ - p.c)*exp(-im*p.α)
         Γ̇v = 1/(2*π*im)*(Γ₊*log.((p.L/2)*ss .- z̃₊) .+  Γ₋*log.((p.L/2)*ss .- z̃₋))/Δt
         #
         ϕ̇₊ .+= real(Γ̇v)
         ϕ̇₋ .+= real(Γ̇v)
         # # # #
         # # # # # add (1/2*\pi*im)*ΓJ*(z\dot -zJ\dot)/(z - zJ)
         żv = zeros(ComplexF64, length(ss))
         # # # # #
         if isempty(ambient_sys) == false
         for (i, source) in enumerate(ambient_sys)
             # żv .+= source.S*Points.cauchy_kernel.(conj.(zss .- source.z)) .*
             #         (ċ .- srcvel[i])
             # żv .+= source.S*conj.(Vortex.Blobs.blob_kernel.(p.L/2*ss .- (source.z-p.c)*exp(-im*p.α), 5e-3)) .*
             #         (ċ .- srcvel[i])*exp(im*p.α)
             żv .+= source.S*conj.(Points.cauchy_kernel.(p.L/2*ss .- (source.z-p.c)*exp(-im*p.α))) .*
                     (ċ .- srcvel[i])*exp(-im*p.α)
         end
         end

         ϕ̇₊ .+= real(żv)
         ϕ̇₋ .+= real(żv)

         return w₊, w₋, ϕ̇₊, ϕ̇₋
end


# We can precompute the tab J^l/l and use it for the complex potential
function ϕtabdot(N, L, A, B₀, B₁, Γ, Jn::Array{ComplexF64,1})
    # Jn[n] = J₊^l/l positive side of the plate
    ϕ̇ = log(Jn[1])*(2Γ/(L*π))

    ϕ̇ += Jn[1] * 2(A[0] - B₀)

    # no need to divide by 2 since Jn[2]= J^2/2
    ϕ̇ += Jn[2] * (A[1] - B₁)

    for n in 2:N-1
        ϕ̇ += A[n]*(-Jn[n-1] + Jn[n+1])
    end

    return 0.25*L*im*ϕ̇
end

function ϕtabdot(N, L, A, B₀, B₁, Γ, J::Array{ComplexF64,2})
    return map( j-> ϕtabdot(N, L, A, B₀, B₁, Γ, J[:,j]), 1:size(J,2))
end


function ϕdot(N, L, A, B₀, B₁, Γ, J::ComplexF64)

    ϕ̇ = log(J)*(2Γ/(L*π))

    ϕ̇ += J * 2(A[0] - B₀)
    ϕ̇ += 0.5*J^2 * (A[1] - B₁)

    Jn = map(l->(1/l)*J^l, 1:N)
    #
    for n in 3:N
        ϕ̇ += A[n-1]*Jn[n]
    end
    #
    for n in 1:N-2
        ϕ̇ -= A[n+1]*Jn[n]
    end
    # for n in 2:N-1
    #     ϕ̇ += A[n]*(J^(n+1)/(n+1) - J^(n-1)/(n-1))
    # end

    return 0.25*L*im*ϕ̇
end

function ϕdot(N, L, A, B₀, B₁, Γ, J::Array{ComplexF64,1})
    return map( j-> ϕdot(N, L, A, B₀, B₁, Γ, j), J)
end


# Compute Jˡ/l for a plate
function Jl_l(idx::Array{Int64,1},N::Int64)
    ss = Plates.Chebyshev.nodes(N)[idx]
    return [(s - im*√(1-s^2))^l/l for  l = 1:N, s in ss]
end

# Compute Jˡ/l for a plate
function Jl_l(ss::Array{Float64,1},N::Int64)
    @assert all(abs.(ss) .<= 1.0) "ss should be in [-0.5, 0.5]"
    return [(s - im*√(1-s^2))^l/l for  l = 1:N, s in ss]
end
