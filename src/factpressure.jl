export rho, J, σ, unitC,
       factstrength, factvorticity_flux,
       factdifftangential,
       factself_induce_velocity,
       factinduce_velocity,
       factCdot,
       factΓdot


rho(ξ) = √(ξ - 1)*√(ξ + 1)

J(ξ) = ξ - rho(ξ)

function σ(n::Int64)
    if n==0
        return 1.0
    elseif n > 0
        return 2.0
    end
end

# withnascent determines whether or not to compute the coefficient for the nascent vortices
# withnascent determines whether or not to compute the coefficient for the nascent vortices
function unitC(p, ambient_sys, t)
    @get p (N, L, c, α, dchebt!)
    C  = zeros(ComplexF64, N)
    Nv = size(ambient_sys, 1)

    n̂ = exp(-im*α)
    Cᵛ = zeros(ComplexF64, N, Nv)

    @inbounds for (j, bj) in enumerate(ambient_sys)
        # put circulation to 1
        Plates.influence_on_plate!(C, p, Vortex.Point(bj.z, 1.0), t)
        view(Cᵛ,:,j) .= C
    end
    return Cᵛ
end

function unitC(p, ambient_sys, z_new, t)
    @get p (N, L, c, α, dchebt!)
    C  = zeros(ComplexF64, N)
    Nv = size(ambient_sys, 1)

    n̂ = exp(-im*α)
    Cᵛ = zeros(ComplexF64, N, Nv+2)

    @inbounds for (j, bj) in enumerate(ambient_sys)
        # put circulation to 1
        Plates.influence_on_plate!(C, p, Vortex.Point(bj.z, 1.0), t)
        view(Cᵛ,:,j) .= C
    end

    # Influence for the new vortex about the leading edge
    Plates.influence_on_plate!(C, p, Vortex.Point(z_new[1], 1.0), t)
    view(Cᵛ,:,Nv+1) .= C

    # Influence for the new vortex about the leading edge
    Plates.influence_on_plate!(C, p, Vortex.Point(z_new[2], 1.0), t)
    view(Cᵛ,:,Nv+2) .= C

    return Cᵛ
end

## γ = aγ⊤Γ + bγ
function factstrength(p::Plate, ṗ, Cᵛ, Γᵛ)
    @get p (N, L, ss, α, B₀, B₁)
    @get ṗ (ċ, c̈, α̇ , α̈ )

    Nv = size(Γᵛ, 1)
    @show Nv
    bγ = zeros(Float64, N)
    aγ = zeros(Float64, N, Nv+2)

    @inbounds for (i, s) in enumerate(ss)
        # Compute bγ
        bγ[i] = -2*B₀*s - B₁*(2s^2-1)
        if abs2(bγ[i]) < 256eps()
            bγ[i] = 0.0
        else
            bγ[i] /= √(1 - s^2)
        end

        # Compute aγ
        U₋ = 1.0
        U  = 2s
        for n in 2:N-1
            for j=1:Nv+2
                aγ[i,j] += 2*imag(Cᵛ[n+1,j])*U
            end
            U₋, U = U, 2s*U - U₋
        end
        aγi = view(aγ,i,:)
        aγi .*= (s^2 - 1)
        aγi .+= -2/(L*π)
        aγi .+= 2*imag.(view(Cᵛ,1,:))*s
        aγi .+= imag.(view(Cᵛ,2,:))*(2*s^2-1)

        for j=1:Nv+2
            if abs2(aγ[i,j]) < 256eps()
                aγ[i,j] = 0.0
            else
                aγ[i,j] /= √(1 - s^2)
            end
        end
    end

    return aγ, bγ
end

## Factorize the difference between mean tangential velocity at z̃
#  and the rigid body motion

function factdifftangential(p::Plate, ṗ, Cᵛ)
    @get p (N, α, ss)
    @get ṗ (ċ, c̈)

    Nv = size(Cᵛ, 2)
    bdiff = -Plates.tangent(ċ, α)*ones(N)
    adiff = zeros(N, Nv)

    offset = 0
    for (i, s) in enumerate(ss)
        -1 ≤ s ≤ 1 || throw(DomainError("s ∉ [-1,1]"))
        T₋ = s
        T  = one(s)

        for i in 1:offset
            T₋, T = T, 2s*T - T₋
        end

        for n in 1+offset:N
            for j=1:Nv
                adiff[i, j] +=  real(Cᵛ[n,j])*T
            end
            T₋, T = T, 2s*T - T₋
        end
    end

    return adiff, bdiff
end

## Build affine map from Γᵛ to δΓ

function factvorticity_flux(p::Plate, Cᵛ, Γᵛ, t, lesp, tesp)

    @get p (N, α, B₀, B₁, L, Γ)

    Nv = size(Γᵛ, 2)

    aδΓ = zeros(2,Nv-2)
    bδΓ = zeros(2)

    view(aδΓ,1,:) .= +2*imag.(Cᵛ[1,1:Nv]) + imag.(Cᵛ[2,1:Nv])
    view(aδΓ,2,:) .= -2*imag.(Cᵛ[1,1:Nv]) + imag.(Cᵛ[2,1:Nv])

    bδΓ[1] = -2B₀ - B₁
    bδΓ[2] = +2B₀ - B₁
    aδΓ .+= -2/(π*L)

    ∂C₁ = view(Cᵛ,:,Nv+1)
    ∂C₂ = view(Cᵛ,:,Nv+2)

    rhs₊ = dot(view(aδΓ,1,:), Γᵛ) + bδΓ[1]
    rhs₋ = dot(view(aδΓ,2,:), Γᵛ) + bδΓ[2]

    # @show rhs₊, rhs₋

    Γ₁ = 1.0
    Γ₂ = 1.0

    A₁₊ =  2imag(∂C₁[1]) + imag(∂C₁[2]) - 2Γ₁/(π*L)
    A₂₊ =  2imag(∂C₂[1]) + imag(∂C₂[2]) - 2Γ₂/(π*L)
    A₁₋ = -2imag(∂C₁[1]) + imag(∂C₁[2]) - 2Γ₁/(π*L)
    A₂₋ = -2imag(∂C₂[1]) + imag(∂C₂[2]) - 2Γ₂/(π*L)

    # @show A₁₊, A₂₊, A₁₋, A₂₋

    if (abs2(lesp) > abs2(rhs₊)) && (abs2(tesp) ≤ abs2(rhs₋))
        view(aδΓ,1,:) .= 0.0
        view(aδΓ,2,:) .*= (-1.0/A₂₋)
        bδΓ .= [0.0; (sign(rhs₋)*tesp - bδΓ[2])/A₂₋]
    elseif (abs2(lesp) ≤ abs2(rhs₊)) && (abs2(tesp) > abs2(rhs₋))
        view(aδΓ,1,:) .*= (-1.0/A₁₊)
        view(aδΓ,2,:) .= 0.0
        bδΓ .= [(sign(rhs₊)*lesp - bδΓ[1])/A₁₊; 0.0]
    elseif (abs2(lesp) > abs2(rhs₊)) && (abs2(tesp) > abs2(rhs₋))
        # No vortex shedding δΓ = [0.0; 0.0]
    else

        detA = A₁₊*A₂₋ - A₂₊*A₁₋
        @assert (detA != 0) "Cannot enforce suction parameters"

        M = 1/detA*[A₂₋ -A₁₋; -A₂₊ A₁₊]

        mul!(aδΓ, M, aδΓ)
        mul!(bδΓ, M, bδΓ)
    end

    return aδΓ, bδΓ
end

## Construct factorization of self_induce_velocity, the self-induced velocity is omitted
# this is equaivalent to the Kirchhoff velocity

function factself_induce_velocity(ambient_sys, t)
    Nv = length(ambient_sys)

    K = zeros(ComplexF64, Nv, Nv)
    for source in 1:Nv, target in source+1:Nv
        δ = √(0.5(ambient_sys[target].δ^2 + ambient_sys[source].δ^2))
        K[target, source] = Vortex.Blobs.blob_kernel(ambient_sys[target].z - ambient_sys[source].z, δ)
        K[source, target] = -K[target, source]
    end
    return K
end

## Construct factorization of induce_velocity of hte bound vortex sheet of the plate
#  onto the vortex elements

function factinduce_velocity(ambient_sys, p::Plate, Cᵛ, t)
    @get p (α, L, c, B₀, B₁)

    Nv = length(ambient_sys)

    ainduce = zeros(ComplexF64, Nv, Nv)
    binduce = zeros(ComplexF64, Nv)

    A₀ = imag.(Cᵛ[1,1:Nv])
    A₁ = imag.(Cᵛ[2,1:Nv])

    @inbounds for (j, bj) in enumerate(ambient_sys)
        ξjstar = conj(2*(bj.z-c)*exp(-im*α)/L)
        ρjstar = rho(ξjstar)
        Jjstar = ξjstar - ρjstar

        ainducej   = view(ainduce,j,:)
        ainducej  .= A₁ .- 2/(π*L)
        ainducej .+= 2*Jjstar*A₀
        rmul!(ainducej, 1/ρjstar)

        binduce[j] = (-2B₀*Jjstar - B₁*Jjstar^2)/ρjstar

        Jⁿ = Jjstar

        @inbounds for n in 2:size(Cᵛ,1)
            ainducej .-= 2imag.(Cᵛ[n,1:Nv])*Jⁿ
            if abs(Jⁿ)<eps()
                break
            end
            Jⁿ *= Jjstar
        end
    end

    rmul!(ainduce, 0.5*im*exp(im*α))
    rmul!(binduce, 0.5*im*exp(im*α))

    return ainduce, binduce
end

## Construct factorization for induce_acc
# Factorize the term $\sum_K \frac{\Gamma_K(conj(w_{-K} - w\tilde_r - i\Omega z\tilde_K))}{2 \pi i (z\tilde - z\tilde_K)^2}
# Ċ_i = aĊ_{ijk} Γ_j Γ_k + bĊ_{ij} Γ_j
# with size(aĊ) = (N, Nv, Nv) and size(bĊ) = (N, Nv)

function factCdot(p::Plate, ṗ, ambient_sys, Cᵛ, t)
    Nv = length(ambient_sys)
    @get p (N, α, L, c, B₀, B₁, zs, dchebt!)
    @get ṗ (ċ, α̇ )

    aĊ = zeros(ComplexF64, N, Nv, Nv)
    bĊ = zeros(ComplexF64, N, Nv)

    K = factself_induce_velocity(ambient_sys, t)
    ainduce, binduce = factinduce_velocity(ambient_sys, p, Cᵛ, t)
    n̂ = exp(-im*α)

    @inbounds for (i, zi) in enumerate(zs)
        for (j, bj) in enumerate(ambient_sys)
            Kdipoleij = Vortex.Points.cauchy_kernel((zi - bj.z)^2)
            view(aĊ,i,j,:) .= Kdipoleij*conj(view(K,j,:) + view(ainduce,j,:))
            bĊ[i,j] = Kdipoleij*(conj(binduce[j]) - conj(ċ))
        end
    end

    rmul!(aĊ, n̂)
    rmul!(bĊ, n̂)

    # Project onto the Chebyshev basis
    tmp = zeros(ComplexF64, N)

    @inbounds for i=1:Nv
        copy!(tmp, view(bĊ,:,i))
        dchebt! * tmp
        view(bĊ,:,i) .= tmp
    end

    @inbounds for i=1:Nv
        for j=1:Nv
        copy!(tmp, view(aĊ,:,i,j))
        dchebt! * tmp
        view(aĊ,:,i,j) .= tmp
        end
    end

    return aĊ, bĊ
end


## Construct quadratic form for the bound_circulation

function factbound_circulation(A, p::Plate, ss::Array{T, 1}) where {T <: Real}
    @get p (N, α, L, c, B₀, B₁, zs)

    Ns = length(ss)
    abound_circ = zeros(Ns, N)
    bbound_circ = zeros(Ns)

    for (i, s) in enumerate(ss)

        abound_circ[i,1] = 2.0
        abound_circ[i,2] = s

        bbound_circ[i] = -2*B₀ - B₁*s
        U₋₂ = 1.0
        U₋  = 2s
        U   = 4s^2 - 1

        for n in 2:N-1
            abound_circ[i, n] = (U/(n+1) - U₋₂/(n-1))
            U₋₂, U₋, U = U₋, U, 2s*U - U₋
        end

        rmul!(view(abound_circ,i,:), -0.5L*√(1 - s^2))
        bbound_circ[i] *= -0.5L*√(1 - s^2)
    end

    # Missing Γterm


    return abound_circ, bbound_circ
end

## Construct quadratic factorization of dΓ/dt

function factΓdot(p::Plate, ṗ, ambient_sys, Cᵛ, Γᵛ, t, lesp, tesp, ss)

    aδΓ, bδΓ = factvorticity_flux(p, Cᵛ, Γᵛ, t, lesp, tesp)

    δΓ = aδΓ*Γᵛ + bδΓ

    aĊ, bĊ = factCdot(p, ṗ, ambient_sys, Cᵛ, t)

    -(Γ₊ + Γ₋)/Δt
    Γ₋/Δt
    Γₛ -= Γ*(acos(s)/π - 1)


end


# Store the unitary A_n coefficients evaluated at the different ξJ locations
# function unitC(p, ambient_sys)
#     @get p (N, L, c, α)
#
#     Nv = size(ambient_sys, 1)
#     Cᵛ = zeros(ComplexF64,N, Nv)
#     fill!(Cᵛ, complex(1.0))
#     @inbounds for (j, bj) in enumerate(blobs)
#         ξj = 2*(bj.z-c)*exp(-im*α)/L
#         ρj = rho(ξj)
#         Jj = ξj - ρj
#         for n=2:N
#             Cᵛ[n,j] = Jj^(n-1)
#         end
#         rmul!(view(Cᵛ,:,j), 1/ρj)
#     end
#
#     rmul!(Cᵛ, σ(1)/(π*L))
#     rmul!(view(Cᵛ,1,:),0.5)
#     return Cᵛ
# end
