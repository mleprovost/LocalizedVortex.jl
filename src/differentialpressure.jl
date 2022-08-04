export strength!, strength, diffstrength!, diffstrength,
       diff_firstkind!, diff_firstkind,
       diag_transform,
       diff_cauchy_kernel, diff_induce_velocity!,
       diff_blob_kernel, _diff_singular_velocity!,
       diff_enforce_no_flow_through!,
       diff_self_induce_velocity!

       #diffpressure

import PotentialFlow.Plates: strength!, strength
using PotentialFlow.Elements

strength(plate, A, Γ) = strength(plate, A, Γ, plate.ss)

function strength(plate, A, Γ, s::Real)
    @get plate (α, B₀, B₁, L, N)

    γ = 0.0

    U₋ = 1.0
    U  = 2s
    for n in 2:N-1
        γ += 2A[n]*U
        U₋, U = U, 2s*U - U₋
    end
    γ *= (s^2 - 1)

    γ += 2Γ/(L*π)
    γ += 2(A[0] - B₀)*s
    γ +=  (A[1] - B₁)*(2s^2 - 1)

    if abs2(γ) < 256eps()
        γ = 0.0
    else
        γ /= √(1 - s^2)
    end
    return γ
end

function strength(plate, A, Γ, ss::AbstractArray{T}) where {T <: Real}
    γs = zeros(Float64, length(ss))
    strength!(γs, plate, A, Γ, ss)
    return γs
end

function strength!(γs, plate, A, Γ, ss = plate.ss)
    for (i, s) in enumerate(ss)
        γs[i] = strength(plate, A, Γ, s)
    end
    nothing
end

function diffstrength!(GA, GΓ, plate, s::Real)
    # GA = zeros(plate.N)
    fill!(GA, 0.0)
    GΓ = -2/(π*plate.c*√(1-s^2))

    # For A0
    GA[1] = 2*s

    # For A1
    GA[2] = (2s^2 - 1)

    # For
    U₋ = 1.0
    U  = 2s
    for n in 2:N-1
        GA[n+1] = 2*U
        U₋, U = U, 2s*U - U₋
    end

    rmul!(GA, 1/√(1 - s^2))

    return GΓ
end

function diffstrength(plate, ss = plate.S)
    JA = zeros(length(ss), plate.N)
    JΓ = zeros(length(ss))-2/(π*plate.c*√(1-s^2))

    @inbounds for (i, s) in enumerate(ss)
        JΓ[i] = diffstrength!(view(JA,i,:), JΓ[i], plate, s)
    end

    return JA, JΓ
end

function diff_firstkind!(GB, s::S, offset = 0) where {S <: Real}
    -1 ≤ s ≤ 1 || throw(DomainError("s ∉ [-1,1]"))
    T₋ = s
    T  = one(S)

    for i in 1:offset
        T₋, T = T, 2s*T - T₋
    end

    N = length(GB)

    for i in 1+offset:N
        GB[i] = T
        T₋, T = T, 2s*T - T₋
    end
end

function diff_firstkind(plate, ss = plate.ss, offset = 0)
    JB = zeros(length(ss), plate.N)

    @inbounds for (i, s) in enumerate(ss)
        diff_firstkind!(view(JB,i,:), s)
    end

    return JB
end

function diag_transform(N)
    D = Diagonal(zeros(ComplexF64, N))

    s = 1/(N-1)

    @inbounds begin
        for n in 2:2:N
            D[n-1] = s
            D[n]   = -s
        end
        if isodd(N)
            D[N] = s
        end

        D[1] /= 2
        D[N] /= 2
    end
    return D
end

# Derivative of the Cauchy kernel computed with respect to z̄
diff_cauchy_kernel(z) = z != zero(z) ? -0.5im/(π*conj(z)^2) : zero(z)


# This function returns the derivative of the non-conjugate velocity kernel
# as used in PotentiaFlow.jl, with respect to the position and strength of the point
function diff_induce_velocity!(G, z::ComplexF64, p::Vortex.Point, t)
    # G[1] = -p.S'*diff_cauchy_kernel(z - p.z)
    # G[2] = -im*G[1]
    # G[3] = PotentialFlow.Points.cauchy_kernel(z - p.z)
    G[3] = PotentialFlow.Points.cauchy_kernel(z - p.z)
    G[1] = -2*π*im*p.S'*G[3]^2
    G[2] = -im*G[1]

    return G
end

# Derivative of the Blob kernel computed with respect to z̄
diff_blob_kernel(z, δ) = z != zero(z) ? -0.5im*(z^2 + δ^2)/(π*(abs2(z) + δ^2)^2) : zero(z)
# diff_blob_kernel(z, δ) = -0.5im*(z^2 + δ^2)/(π*(abs2(z) + δ^2)^2)

# This function returns the derivative of the non-conjugate velocity kernel
# as used in PotentiaFlow.jl, with respect to the position and strength of the point
function diff_induce_velocity!(G, z::ComplexF64, p::Vortex.Blob, t)
    # This computation assumes that all the Blob have the ame regularization parameter δ
    G[1] = -p.S'*diff_blob_kernel(z - p.z, p.δ)
    G[2] = -im*G[1]
    G[3] = PotentialFlow.Blobs.blob_kernel(z - p.z, p.δ)
    return G
end

function diff_induce_velocity!(G, zz::Array{ComplexF64,1}, p::Vortex.Point, t)
    @assert size(G) == (size(zz,1), 3)
    for (i, zi) in enumerate(zz)
        diff_induce_velocity!(view(G, i, :), zi, p, t)
    end
end

function diff_induce_velocity!(G, zz::Array{ComplexF64,1}, b::Vortex.Blob, t)
    @assert size(G) == (size(zz,1), 3)
    for (i, zi) in enumerate(zz)
        diff_induce_velocity!(view(G, i,), zi, b, t)
    end
end



function diff_induce_velocity!(J, p::Plate, sources::T, t) where T <: Union{Tuple, AbstractArray}
    for (i, source) in enumerate(sources)
        diff_induce_velocity!(view(J,:,3*i-2:3*i), p, source, t)
    end
    J
end

function diff_induce_velocity!(G, p::Plate, src, t)
    _diff_singular_velocity!(G, p, Elements.unwrap(src), t,
                             kind(Elements.unwrap_src(src)))
end

function _diff_singular_velocity!(G, p, src::Vortex.Blob, t, ::Type{PotentialFlow.Elements.Singleton})
    diff_induce_velocity!(G, p.zs, Vortex.Point(src.z, src.S), t)
end

# This function returns the Jacobian of p.C and p.Γ with respect to the state components
function diff_enforce_no_flow_through!(JC, p::Plate, ṗ, elements, t)
    @get p (L, C, α, dchebt!)
    @get ṗ (ċ, α̇)

    fill!(C, zero(ComplexF64))

    diff_induce_velocity!(JC, p, elements, t)

    n̂ = exp(-im*α)
    rmul!(JC, n̂)

    @inbounds for col in eachcol(JC)
        Plates.Chebyshev.transform!(col, dchebt!.dct!)
        # JC[:,i] .= dchebt! * JC[:,i]
    end
    # JΓ = zeros(3*length(elements), 1)
    # for i=1:length(elements)
    #     JΓ[3*(i-1)+] = -1.0
    # end
    return JC#, JΓ
end


function diff_self_induce_velocity!(J, blobs::Vector{Vortex.Blob}, t)
    N = length(blobs)

    for s in 1:N, t in 1#:N
            δ = √(0.5(blobs[t].δ^2 + blobs[s].δ^2))
            K = PotentialFlow.Blobs.blob_kernel(blobs[t].z - blobs[s].z, δ)
            diff_K = diff_blob_kernel(blobs[t].z - blobs[s].z, δ)
            @show K, diff_K
            # Jacobian with respect to the source
            J[t, 3*s-2] += -blobs[s].S'*diff_K
            J[t, 3*s-1] += -im*J[t, 3*s-2]
            J[t, 3*s]   += K

            # Use symmetry of the Kernel
            # J[s, 3*t-2] += -blobs[t].S'*diff_K
            # J[s, 3*t-1] += -im*J[s, 3*t-2]
            # J[s, 3*t]   += K
        # J[s] -= blobs[t].S'*K
    end
    J
end

# This function differentiate the induced velocity on the vortex elements of the plate
# D v_{plate -> blobs} = ∂v_{plate -> blobs}/∂C ∂C/∂li + ∂v_{plate -> blobs}/∂Γ ∂Γ/∂li + ∂v_{plate -> blobs}/∂li
# ∂v_{plate -> blobs}/∂A  is called GA
# ∂v_{plate -> blobs}/∂Γ  is called GΓ
# ∂v_{plate -> blobs}/∂li is called G

function diff_induce_velocity!(GA, GΓ, G, z::ComplexF64, p::Plate, t)
    @get p (α, L, c, B₀, B₁, Γ, A)

    fill!(GA, complex(0.0))
    fill!(GΓ, complex(0.0))
    fill!(G,  complex(0.0))

    z̃ = conj(2*(z - c)*exp(-im*α)/L)

    ρ = √(z̃ - 1)*√(z̃ + 1)
    J = z̃ - ρ

    # w = (A[1] + 2Γ/(π*L))
    # w += 2(A[0] - B₀)*J
    # w -= B₁*J^2
    #
    # w /= ρ

    GΓ[1] = 0.5im*exp(im*α)*2/(π*L)

     # Term A0
    GA[1] =  2*J/ρ

    # Term A1
    GA[2] = 1.0/ρ
    #

    # Higher terms, note the shift between A that starts at index 0
    # and GA that starts at index 1
    Jⁿ = J
    for n in 1:length(A)-1
        GA[n+1] -= 2*Jⁿ
        Jⁿ *= J
    end

    rmul!(GA, 0.5*im*exp(im*α))

      # note that this provides u+iv (JDE)
    # 0.5im*w*exp(im*α)

end


# function diff_transform(C::Transform{T, true}, A::Vector{T}) where {T}
#     N = length(A)
#     D = diag_transform()
#
#
#
#
#
#
# end




function diffpressure(p::Plate, ṗ, ambient_sys, z_new, t, Δt, lesp, tesp)
    blobs, lesp, tesp = state_to_blobs(state, config.δ)
    plate = Plate(config.N, config.L, complex(t + config.Δt), config.α)
    motion = Plates.RigidBodyMotion(config.ċ, 0.0)

    z₊ = (blobs[end-1].z + 2plate.zs[end])/3
    z₋ = (blobs[end].z + 2plate.zs[1])/3

    # Code to enforce no-flow through
        C = zeros(ComplexF64. p.N)

        C = induce_velocity(p, ambient_sys, t)

        n̂ = exp(-im*α)
        rmul!(C, n̂)

        dchebt = Chebyshev.plan_transform(zero(C))
        C = dchebt*C

        p.Γ = -circulation(elements)
        p.B₀ = normal(ṗ.ċ, p.α)
        p.B₁ = 0.5*ṗ.α̇ *L
    # Code to compute the induce velocity on all the elements
        srcvel = self_induce_velocity(ambient_sys, t)
        @show srcvel
        induce_velocity!(srcvel, ambient_sys, p, t)

        targvel = fill(ċ, length(p))
        Ċ = zero(targvel)

        induce_acc!(Ċ, p.zs, targvel, ambient_sys, srcvel)

end
