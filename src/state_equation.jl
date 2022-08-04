export state_equation2

function state_equation2(state, t, config)
    blobs, lesp, tesp = state_to_blobs(state, config.δ)
    plate = Plate(config.N, config.L, complex(t), config.α)

    sys = (plate, blobs)
    motion = Plates.RigidBodyMotion(config.ċ, 0.0)

    shed_new_vorticity!(blobs, plate, motion, config, t, lesp, tesp)

    ẋs = (motion, allocate_velocity(blobs))
    self_induce_velocity!(ẋs, sys, t)

    if config.transfer && (length(blobs) > 20)
        ΔZ = pairwise_corrections(blobs, plate, 10)
    end

    advect!(sys, sys, ẋs, config.Δt)

    if config.transfer && (length(blobs) > 20)
        err = zeros(ComplexF64, size(ΔZ))
        transfer_error!(err, ΔZ, blobs, plate)
        err += I
	# 	if inds == nothing
    #     	inds = collect(Iterators.filter(i -> abs(err[i]) < ϵaggregation1, CartesianIndices(size(err))))
	# 	end
    #     transfer_circulation!(blobs, plate, ΔZ, err, inds, ϵaggregation2)
		return blobs_to_state(sys[2], lesp, tesp), ΔZ, err
	else
		return blobs_to_state(sys[2], lesp, tesp)
	end
end
