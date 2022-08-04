using PotentialFlow
import PotentialFlow.Plates: Plate
import PotentialFlow.Utils: @get

export matrix_induce_velocity_blobs,
       matrix_self_induce_velocity_blobs,
       matrix_induce_velocity_singular,
       strength_motion,
       strength_motion!,
       matrix_strength_vortex,
       matrix_suction_parameters,
       matrix_impulse_vortex,
       linearsystem_aggregation

function matrix_induce_velocity_blobs(target::Array{ComplexF64, 1}, source::Array{PotentialFlow.Blobs.Blob{Float64,Float64},1}, t)
    A = zeros(ComplexF64, size(target, 1), size(source, 1))
    for i=1:size(target,1)
        for j=1:size(source,1)
        A[i,j] = PotentialFlow.Blobs.blob_kernel(target[i]-source[j].z, Elements.blobradius(source[j]))
        end
    end
    return A
end

# matrix_self_induce_velocity_blobs(blobs,t)*Γ-self_induce_velocity(blobs, t)

function matrix_self_induce_velocity_blobs(source::Array{PotentialFlow.Blobs.Blob{Float64,Float64},1}, t)
    A = zeros(ComplexF64, size(source, 1), size(source, 1))
    for i=1:size(source,1)
        for j=i:size(source,1)
        A[i,j] = PotentialFlow.Blobs.blob_kernel(source[i].z-source[j].z, Elements.blobradius(source[j]))
        A[j,i] = -A[i,j]
        end
    end
    return A
end

# matrix_induce_velocity_blobs(plate.zs, blobs,t)*Γ-induce_velocity(plate.zs, blobs, t)

function matrix_induce_velocity_singular(target::Array{ComplexF64, 1}, source::Array{PotentialFlow.Blobs.Blob{Float64,Float64},1}, t)
    A = zeros(ComplexF64, size(target, 1), size(source, 1))
    for i=1:size(target,1)
        for j=1:size(source,1)
        A[i,j] = PotentialFlow.Points.cauchy_kernel(target[i]-source[j].z)
        end
    end
    return A
end

# matrix_induce_velocity_singular(plate.zs, blobs, t)*Γ-induce_velocity(plate, blobs, t)

# Compute the strength induced only by the motion of the plate
function strength_motion(plate::Plate{T}, s::Real) where {T}
    @get plate (α, B₀, B₁, L, Γ, N)

    γ = zero(T)

    γ -= 2*B₀*s
    γ -=  B₁*(2s^2 - 1)

    if abs2(γ) < 256eps()
        γ = 0.0
    else
        γ /= √(1 - s^2)
    end
    return γ
end

function strength_motion(plate::Plate{R}, ss::AbstractArray{T}) where {R, T <: Real}
    γs = zeros(R, length(ss))
    strength_motion!(γs, plate, ss)
    return γs
end

function strength_motion!(γs, plate, ss = plate.ss)
    for (i, s) in enumerate(ss)
        γs[i] = strength_motion(plate, s)
    end
    nothing
end

strength_motion(plate) = strength_motion(plate, plate.ss)

function matrix_strength_vortex(p::Plate{T}, source::Array{PotentialFlow.Blobs.Blob{Float64,Float64},1}, t, ss::AbstractArray{S}) where {T, S}

    A = zeros(size(ss,1), size(source, 1))
    K = @. inv(sqrt(1 - ss^2))
    rmul!(K, -1/(π*p.L))

   @fastmath @inbounds for (j, srcj) in enumerate(source)
        sj = (srcj.z-p.c)*exp(-im*p.α)
        sj *= 2/p.L
        Ksrcj = √(sj-1.0)*√(sj+1.0)
        for i=1:size(ss,1)
        # K = -1/(π*p.L*(1.0 - ss[i]^2)^(0.5))
        # Eq 8.53 is expressed in terms of the scaled coordinate.

        A[i,j] = 2*K[i]*real(Ksrcj/(sj - ss[i]))
        #         A[i,j] = real(K*((√(sj-1.0)*√(sj+1.0))/(sj - ss[i]) +
        #                   (√(conj(sj)-1.0)*√(conj(sj)+1.0))/(conj(sj) - conj(ss[i]))))
        end
    end
    return A
end

matrix_strength_vortex(p::Plate{T}, source::Array{PotentialFlow.Blobs.Blob{Float64,Float64},1}, t) where {T} =  matrix_strength_vortex(p, source, t, p.ss)

function matrix_suction_parameters(p::Plate{T}, source::Array{PotentialFlow.Blobs.Blob{Float64,Float64},1}) where {T}
    A = zeros(2, size(source,1))
    @inbounds for (i, srci) in enumerate(source)
        si  = (srci.z-p.c)*exp(-im*p.α)
        si  *= 2/p.L
        si₊ = sqrt(si + 1.0)
        si₋ = sqrt(si - 1.0)
        A[1,i] = real(si₊/si₋)
        A[2,i] = real(si₋/si₊)
    end
    rmul!(A, 1/(2*π*p.L))
    rmul!(A, -4.0) # to match results in PotentialFlow.jl
    return A
end

function matrix_impulse_vortex(source::Array{PotentialFlow.Blobs.Blob{Float64,Float64},1}) where {T}
    A = zeros(2, size(source,1))
    @inbounds for (i, srci) in enumerate(source)
        si  = (srci.z-p.c)*exp(-im*p.α)
        si  *= 2/p.L
        si₊ = sqrt(si + 1.0)
        si₋ = sqrt(si - 1.0)
        A[1,i] = -imag(si)
        A[2,j] = -real(si) - real(si₋*si₊ - si)
    end
    return A
end

function linearsystem_aggregation(p::Plate{T}, source::Array{PotentialFlow.Blobs.Blob{Float64,Float64},1}, t, ss::AbstractArray{S}) where {T, S}
    nsource = size(source,1)
    ntarget = size(ss,1)
    A = zeros(nsource*2 + ntarget*3 + 2 + 1, nsource)

    # Self-induced velocity between the blobs
    @inbounds for i=1:size(source,1)
        for j=i+1:size(source,1)
        vij = PotentialFlow.Blobs.blob_kernel(source[i].z-source[j].z, Elements.blobradius(source[j]))
        A[i,j] = real(vij)
        A[nsource+i,j] = imag(vij)
        A[j,i] = -A[i,j]
        A[nsource+j,i] = -A[nsource+i,j]
        end
    end

    K = @. inv(sqrt(1 - ss^2))
    rmul!(K, -1/(π*p.L))

    @fastmath @inbounds for (j, srcj) in enumerate(source)
        sj = (srcj.z-p.c)*exp(-im*p.α)
        sj *= 2/p.L
        sj₊ = √(sj + 1)
        sj₋ = √(sj - 1)
        for i=1:size(ss,1)

        # Induced velocity on a set of target locations
        if abs2(sj-ss[i])<10*eps()
            Cij = complex(0.0)
        else
            Cij = 1/(sj-ss[i])
        end
        vij = (0.5*im/π)*conj(-Cij)
#         K = -1/(π*p.L*(1.0 - ss[i]^2)^(0.5))
        # Eq 8.53 is expressed in terms of the scaled coordinate.
        A[2*nsource+i,j] = real(vij)
        A[2*nsource+ntarget+i,j] = imag(vij)
        A[2*nsource+2*ntarget+i,j] = 2*K[i]*real((sj₊*sj₋)*Cij)
#         A[i,j] = real(K*((√(sj-1.0)*√(sj+1.0))/(sj - ss[i]) +
#                     (√(conj(sj)-1.0)*√(conj(sj)+1.0))/(conj(sj) - conj(ss[i]))))
        end
        # Individual contribution to the LESP
        A[end-2,j] = -2/(π*p.L)*real(sj₊/sj₋)
        # Individual contribution to the TESP
        A[end-1,j] = -2/(π*p.L)*real(sj₋/sj₊)
    end

    # Individual contribution to the total circulation
    A[end,:] .= 1.0

    return A, A*circulation.(source)
end
