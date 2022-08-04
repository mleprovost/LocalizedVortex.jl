using PotentialFlow
import PotentialFlow.Plates: Plate
import PotentialFlow.Utils: @get

export VortexParams,
       impulse_matching_correction, total_impulse, pairwise_corrections,
       transfer_error!, transfer_error, shed_new_vorticity!,
       state_equation, mean_state_equation, new_state_equation, state_to_blobs, blobs_to_state,
       regen_forces, medianfilter1, medianfilter,
       cleansignal!, cleansignal



struct VortexParams
	# Vortex Parameters
    N::Int
    L::Float64
    ċ::ComplexF64
    α::Float64
    δ::Float64
    Δt::Float64
    transfer::Bool

	# Critical TESP
	tesp::Float64

	# Additive inflation
	ϵX::Float64
	ϵΓ::Float64
	ϵLESP::Float64

	# Multiplicative Inflation
	β::Float64

	# Measurement noise
	ϵY::Float64
end

function impulse_matching_correction(vₛ, vₜ, plate::Plate)
    @get plate (c, α, L)

    c⁺ = plate.zs[end]
    c⁻ = plate.zs[1]

    z̃ₜ = 2(vₜ.z - c)*exp(-im*α)/L
    z̃ₛ = 2(vₛ.z - c)*exp(-im*α)/L

    pₜ = Plates.unit_impulse(z̃ₜ)
    pₛ = Plates.unit_impulse(z̃ₛ)

    η = z̃ₜ/(√(z̃ₜ + 1)*√(z̃ₜ - 1))

    Δp = pₛ - pₜ
    Δz = im*0.5L*exp(im*α)*(Δp*(1 + η') + Δp'*(η' - 1))/(η + η')*(circulation(vₛ)/circulation(vₜ))
    return Δz
end

function total_impulse(point, plate)
    @get plate (L, c, α)
    z = Elements.position(point)
    Γ = circulation(point)

    z̃ = 2exp(-im*α)*(z - c)/L
    exp(im*α)*0.5L*Γ*(imag(z) - im*real(√(z - 1)*√(z + 1)))
    # exp(im*α)*0.5L*Γ*(imag(z̃) - im*real(√(z̃ - 1)*√(z̃ + 1)))
end

function pairwise_corrections(blobs, plate, n=20)
    [impulse_matching_correction(blobs[j], blobs[i], plate) for i in 1:length(blobs)-n, j in 1:length(blobs)-n]
end

function transfer_error!(err, ΔZ, blobs, plate)
    for s in 1:size(err,2), t in 1:size(err,1)
        if s == t
            continue
        end
        err[t,s] = let bₜ = blobs[t],
                       bₛ = blobs[s],
                       Δz = ΔZ[t,s],
                       pₜ = total_impulse(Vortex.Point(bₜ.z + Δz, circulation((bₜ, bₛ))), plate),
                       pₛ = total_impulse(bₛ, plate) + total_impulse(bₜ, plate)
            pₜ - pₛ
        end
    end
    err
end

transfer_error(ΔZ, blobs, plate) = transfer_error!(zeros(ComplexF64, size(ΔZ)), ΔZ, blobs, plate)

function transfer_circulation!(blobs, plate, ΔZ, err, inds, ϵ = 1e-4)
    used_target = Set{Int}()
    Σerr = 0.0
#     p = sortperm([abs2(err[i]) for i in inds])
    for ind in inds
        tind, sind = ind.I
        if (circulation(blobs[sind]) == 0) || (tind ∈ used_target)
            continue
        end
        if (Σerr + abs(err[tind, sind])) > ϵ
            break
        end
        Σerr += abs(err[tind, sind])
        push!(used_target, tind)
        Δz = ΔZ[tind, sind]
        blobs[tind] = Vortex.Blob(blobs[tind].z + Δz, circulation(blobs[tind]) + circulation(blobs[sind]), blobs[tind].δ)
        blobs[sind] = Vortex.Blob(blobs[sind].z, 0, blobs[sind].δ)
    end
end

function transfer_circulation(blobs, plate, ΔZ, err, inds, ϵ = 1e-4)
    used_target = Set{Int}()
    Σerr = 0.0
	# @show inds
#     p = sortperm([abs2(err[i]) for i in inds])
    for (i, ind) in enumerate(inds)
        tind, sind = ind.I
        if (circulation(blobs[sind]) == 0) || (tind ∈ used_target)
            continue
        end
        if (Σerr + abs(err[tind, sind])) > ϵ
			return i-1
            break
        end
        Σerr += abs(err[tind, sind])
        push!(used_target, tind)
        Δz = ΔZ[tind, sind]
        blobs[tind] = Vortex.Blob(blobs[tind].z + Δz, circulation(blobs[tind]) + circulation(blobs[sind]), blobs[tind].δ)
        blobs[sind] = Vortex.Blob(blobs[sind].z, 0, blobs[sind].δ)
    end

end

function shed_new_vorticity!(blobs, plate, motion, config, t, lesp = 0.0, tesp = 0.0)
     # if isempty(blobs)
          Δz₊ = Δz₋ = config.Δt
		  z₊ = plate.zs[end] + im*Δz₊*exp(im*plate.α)
		  z₋ = plate.zs[1]   -    Δz₋*exp(im*plate.α)
     # else
		#  z₊ = (blobs[end-1].z + 2plate.zs[end])/3
		#  z₋ = (blobs[end].z + 2plate.zs[1])/3
         # Δz₊ = min(minimum(abs, (Vortex.position(blobs) .- 2plate.zs[end])/3), config.Δt)
         # Δz₋ = min(minimum(abs, (Vortex.position(blobs) .- 2plate.zs[1])/3), config.Δt)
     # end



    blob₊ = Vortex.Blob(z₊, 1.0, config.δ)
    blob₋ = Vortex.Blob(z₋, 1.0, config.δ)
    Plates.enforce_no_flow_through!(plate, motion, blobs, t)

    Γ₊, Γ₋, _, _ = Plates.vorticity_flux!(plate, blob₊, blob₋, t, lesp, tesp);
	# @show Γ₊, Γ₋

    push!(blobs, Vortex.Blob(z₊, Γ₊, config.δ), Vortex.Blob(z₋, Γ₋, config.δ))
end

# function state_equation(state, t, config; ϵaggregation1::Float64 = 2e-4, ϵaggregation2::Float64 = 1e-3)

function state_equation(state, t, config; ϵaggregation1::Float64 = 3e-4, ϵaggregation2::Float64 = 8e-4, inds = nothing)
    blobs, lesp, tesp = state_to_blobs(state, config.δ)
    plate = Plate(config.N, config.L, complex(t), config.α)

    sys = (plate, blobs)
    motion = Plates.RigidBodyMotion(config.ċ, 0.0)
#     ẋs = (motion, allocate_velocity(blobs))
# #     Filter out points too close to the plate
#     map!(blobs, blobs) do b
#         z = exp(-im*α)*(b.z - plate.c)
# #         if (abs(real(z)) < 0.5L + 0.2Δt) && (abs(imag(z)) < 2.5Δt)
#         if abs(real(z)) < 0.5L
#             H = Vortex.Plates.smootherstep(1Δt, 5Δt, abs(imag(z)))
#             Γ = H*b.Γ
#             Vortex.Blob(b.z, Γ, b.δ)
#         else
#             b
#         end
#     end
    shed_new_vorticity!(blobs, plate, motion, config, t, lesp, tesp)
	# @show blobs
	# @show typeof(blobs) <: Vector{PotentialFlow.Blobs.Blob{T,R}} where {T,R}
    ẋs = (motion, allocate_velocity(blobs))
    self_induce_velocity!(ẋs, sys, t)

    if config.transfer && (length(blobs) > 20)
        ΔZ = pairwise_corrections(blobs, plate, 10)
    end

    advect!(sys, sys, ẋs, config.Δt)

    if config.transfer && (length(blobs) > 20)
        err = zeros(ComplexF64, size(ΔZ))
        transfer_error!(err, ΔZ, blobs, plate)
        # @show minimum(abs.(err[abs.(err) .> 0.0]))
        # @show maximum(abs.(err[abs.(err) .> 0.0]))
        err += I
		if inds == nothing
        	inds = collect(Iterators.filter(i -> abs(err[i]) < ϵaggregation1, CartesianIndices(size(err))))
		end
        transfer_circulation!(blobs, plate, ΔZ, err, inds, ϵaggregation2)
    end
    blobs_to_state(sys[2], lesp, tesp)
end

function mean_state_equation(state, t, config; ϵaggregation1::Float64 = 3e-4, ϵaggregation2::Float64 = 8e-4)
    blobs, lesp, tesp = state_to_blobs(state, config.δ)
    plate = Plate(config.N, config.L, complex(t), config.α)

    sys = (plate, blobs)
    motion = Plates.RigidBodyMotion(config.ċ, 0.0)

    shed_new_vorticity!(blobs, plate, motion, config, t, lesp, tesp)
	# @show blobs
	# @show typeof(blobs) <: Vector{PotentialFlow.Blobs.Blob{T,R}} where {T,R}
    ẋs = (motion, allocate_velocity(blobs))
    self_induce_velocity!(ẋs, sys, t)

    if config.transfer && (length(blobs) > 20)
        ΔZ = pairwise_corrections(blobs, plate, 10)
    end

    advect!(sys, sys, ẋs, config.Δt)

    if config.transfer && (length(blobs) > 20)
        err = zeros(ComplexF64, size(ΔZ))
        transfer_error!(err, ΔZ, blobs, plate)
        # @show minimum(abs.(err[abs.(err) .> 0.0]))
        # @show maximum(abs.(err[abs.(err) .> 0.0]))
        err += I
        inds = collect(Iterators.filter(i -> abs(err[i]) < ϵaggregation1, CartesianIndices(size(err))))
        lastidx = transfer_circulation(blobs, plate, ΔZ, err, inds, ϵaggregation2)
		if lastidx != nothing
			inds = inds[1:lastidx]
		end
		return blobs_to_state(sys[2], lesp, tesp), inds
    else
		return blobs_to_state(sys[2], lesp, tesp), nothing
	end
end

function new_state_equation(state, t, config; ϵaggregation::Float64 = 3e-3)#2.6e-3)
    blobs, lesp, tesp = state_to_blobs(state, config.δ)
    plate = Plate(config.N, config.L, complex(t), config.α)

    sys = (plate, blobs)
    motion = Plates.RigidBodyMotion(config.ċ, 0.0)

    shed_new_vorticity!(blobs, plate, motion, config, t, lesp, tesp)
    ẋs = (motion, allocate_velocity(blobs))
    self_induce_velocity!(ẋs, sys, t)

    advect!(sys, sys, ẋs, config.Δt)

	if config.transfer && (length(sys[2]) > 10)
		# From the linear system
		A, b = linearsystem_aggregation(sys[1], sys[2], t + config.Δt, sys[1].ss[4:10:end-4])
		# measure_state()
		# Don't touch the last ten vortices
		# b .-= A[:,end-9:end]*circulation.(sys[2][end-9:end])
		# A = A[:,1:end-10]
		# Use the QR OMP to identify the sparse set of vortices
		idxqr, cqr, ϵqr = qromp(A, b; ϵrel = ϵaggregation, maxterms = 26)#, keepidx = [length(sys[2])-1; length(sys)[2]])#maxterms = length(sys[2])-2)#, keepidx = collect(length(sys[2])-10:length(sys[2])))
		@show ϵqr[end]./norm(b)
		sortpermidxqr = sortperm(idxqr)
		sortidxqr = sort(idxqr)
		nb = size(idxqr, 1)
		@show nb

		# blobs = nb > 0 ? map(Vortex.Blob, zs, Γs, δs) : Vortex.Blob{Float64, Float64}[]
		blobs = PotentialFlow.Blobs.Blob{Float64,Float64}[]
		for i=1:nb
			push!(blobs, Vortex.Blob(sys[2][sortidxqr[i]].z, cqr[sortpermidxqr[i]], config.Δt))
		end

		# for i=size(sys[2],1)-9:size(sys[2],1)
		# 	push!(blobs, sys[2][i])
		# end

	    blobs_to_state(blobs, lesp, tesp)
	else

		blobs_to_state(sys[2], lesp, tesp)
	end
end


function state_to_blobs(states, δ)
    N = (length(states)-2)÷3
    zs = (states[3i-2] + im*states[3i-1] for i in 1:N)
    Γs = (states[3i] for i in 1:N)
    δs = (δ for i in 1:N)

    blobs = N > 0 ? map(Vortex.Blob, zs, Γs, δs) : Vortex.Blob{Float64, Float64}[]

    blobs, states[end-1], states[end]
end

function blobs_to_state(blobs, lesp, tesp)
    states = Vector{Float64}(undef, 3*length(blobs) + 2)
    for (i, b) in enumerate(blobs)
        states[3i-2] = real(b.z)
        states[3i-1] = imag(b.z)
        states[3i]   = circulation(b)
    end
    states[end-1] = lesp
    states[end] = tesp
    states
end


function regen_forces(T, states, config)
    forces = Vector{ComplexF64}(undef, length(states))
    for (i, state) in enumerate(states[2:end])
        blobs, lesp, tesp = state_to_blobs(state, config.δ)

        # plate = Plate(config.N, config.L, complex(config.Δt*(i-1)), config.α)
		plate = Plate(config.N, config.L, complex(T[i]), config.α)
        motion = Plates.RigidBodyMotion(config.ċ, 0)
        shed_new_vorticity!(blobs, plate, motion, config, T[i], lesp, tesp)
        vels = allocate_velocity(blobs)
		induce_velocity!(vels, blobs, plate, T[i])
        self_induce_velocity!(vels, blobs, T[i])
        forces[i] = Plates.force(plate, motion, blobs, vels, (blobs[end-1], blobs[end-2]), config.Δt)
        # forces[i] = Plates.force(plate, motion, blobs, vels, (blobs[end], blobs[end-1]), config.Δt)
    end
    forces
end


#### Median filter tools
medianfilter1(v,ws) = [median(v[i:(i+ws-1)]) for i=1:(length(v)-ws+1)]
medianfilter(v) = medianfilter1(vcat(0,v,0), 7)
cleansignal!(v,ws) = vcat(v[1:ws-1], medianfilter1(v,ws))
cleansignal(ws) = v -> cleansignal!(v, ws)
