export  finitediff_pressure

function finitediff_pressure(blobs::Vector{Vortex.Blob{Float64, Float64}}, t, Δt, lesp, tesp, config, idxp = 1:config.N)
    N = config.N
    L = config.L
    α = config.α
    c = complex(t)
    δ = config.δ
    ċ = config.ċ

    blobsx⁺ = deepcopy(blobs)
    blobsy⁺ = deepcopy(blobs)
    blobsΓ⁺ = deepcopy(blobs)

    # Define the different plates
    p = PotentialFlow.Plate(128,L,c,α)
    px⁺ = deepcopy(p)
    py⁺ = deepcopy(p)
    pΓ⁺ = deepcopy(p)
    plesp⁺ = deepcopy(p)

    motion = PotentialFlow.RigidBodyMotion(ċ, 0.0)

    DELTA=1e-6

    Nblob = length(blobs)

    z₊ = (blobs[end-1].z + 2p.zs[end])/3
    z₋ = (blobs[end].z + 2p.zs[1])/3

    ss = Plates.Chebyshev.nodes(config.N)[idxp]

    Ny = length(idxp)
    # Nblob+1 since we differentiate also with respect to the LESP value
    J = zeros(Ny, 3*Nblob+1)

    zblob = Elements.position(blobs)
    Γblob = Elements.circulation.(blobs)

    # Plates.enforce_no_flow_through!(p, motion, blobs, t)
    press_fd = Plates.surface_pressure_inst(p,motion,blobs,(z₊,z₋),t,Δt,lesp,tesp,ss)

    dz = DELTA
    dΓ = DELTA
    dlesp = DELTA

    @inbounds for i=1:Nblob

        blobsx⁺[i] = Vortex.Blob(zblob[i]+complex(dz),Γblob[i],δ)
        blobsy⁺[i] = Vortex.Blob(zblob[i]+im*dz,Γblob[i],δ)
        blobsΓ⁺[i] = Vortex.Blob(zblob[i],Γblob[i]+dΓ,δ)

        # Plates.enforce_no_flow_through!(p,   motion, blobs, t)
        # Plates.enforce_no_flow_through!(px⁺, motion, blobsx⁺, t)
        # Plates.enforce_no_flow_through!(py⁺, motion, blobsy⁺, t)
        # Plates.enforce_no_flow_through!(pΓ⁺, motion, blobsΓ⁺, t)

        pressx⁺_fd = Plates.surface_pressure_inst(px⁺,motion,blobsx⁺,(z₊,z₋),t,Δt,lesp,tesp,ss)
        pressy⁺_fd = Plates.surface_pressure_inst(py⁺,motion,blobsy⁺,(z₊,z₋),t,Δt,lesp,tesp,ss)
        pressΓ⁺_fd = Plates.surface_pressure_inst(pΓ⁺,motion,blobsΓ⁺,(z₊,z₋),t,Δt,lesp,tesp,ss)

        dpdx_fd = (pressx⁺_fd - press_fd)/dz
        dpdy_fd = (pressy⁺_fd - press_fd)/dz
        # dpdz_fd = 0.5*(dpdx_fd - im*dpdy_fd)
        # dpdzstar_fd = 0.5*(dpdx_fd + im*dpdy_fd)
        dpdΓ_fd = (pressΓ⁺_fd - press_fd)/dΓ

        J[:,3i-2] .= dpdx_fd
        J[:,3i-1] .= dpdy_fd
        J[:,3i] .= dpdΓ_fd

        blobsx⁺[i] = blobs[i]
        blobsy⁺[i] = blobs[i]
        blobsΓ⁺[i] = blobs[i]
    end

    # Plates.enforce_no_flow_through!(plesp⁺, motion, blobs, t)
    presslesp⁺_fd = Plates.surface_pressure_inst(plesp⁺,motion,blobs,(z₊,z₋),t,Δt,lesp+dlesp,tesp,ss)
    dpdlesp_fd = (presslesp⁺_fd - press_fd)/dlesp

    J[:,3*Nblob+1] .= dpdlesp_fd
    return J
end

# function finitediff_pressure(blobs, t, Δt, lesp, tesp, config, idxp = 1:config.N)
#
#     L = config.L
#     α = config.α
#     δ = config.δ
#     c = complex(t)
#     ċ = config.ċ
#
#     ss = Plates.Chebyshev.nodes(config.N)[idxp]
#
#     blobs⁺ = deepcopy(blobs)
#
#     # Define the different plates
#     p = PotentialFlow.Plate(128,L,c,α)
#     p⁺ = deepcopy(p)
#
#     motion = PotentialFlow.RigidBodyMotion(ċ, 0.0)
#
#     DELTA = 1e-6
#
#     Nblob = length(blobs)
#
#     z₊ = (blobs[end-1].z + 2p.zs[end])/3
#     z₋ = (blobs[end].z + 2p.zs[1])/3
#
#     Ny = length(ss)
#     # Nblob+1 since we differentiate also with respect to the LESP value
#     J = zeros(Ny, 3*Nblob+1)
#     # fill!(J, 0.0)
#     zblob = Elements.position(blobs)
#     Γblob = Elements.circulation.(blobs)
#
#     Plates.enforce_no_flow_through!(p, motion, blobs, t)
#     # @show PotentialFlow.Plates.suction_parameters(p)
#     press_fd = Plates.surface_pressure_inst(p,motion,blobs,(z₊,z₋),t,Δt,lesp,tesp,ss)
#
#     dz = DELTA
#     dΓ = DELTA
#     dlesp = DELTA
#
#     press⁺_fd = zeros(Ny)
#
#     for i=1:Nblob
#
#         blobs⁺[i] = Vortex.Blob(zblob[i]+dz,Γblob[i],δ)
#         press⁺_fd .= Plates.surface_pressure_inst(p⁺,motion,blobs⁺,(z₊,z₋),t,Δt,lesp,tesp,ss)
#         J[:,3i-2] .= (1.0/dz)*(press⁺_fd - press_fd)
#
#         blobs⁺[i] = Vortex.Blob(zblob[i]+im*dz,Γblob[i],δ)
#         press⁺_fd .= Plates.surface_pressure_inst(p⁺,motion,blobs⁺,(z₊,z₋),t,Δt,lesp,tesp,ss)
#         J[:,3i-1] .= (1.0/dz)*(press⁺_fd - press_fd)
#
#         blobs⁺[i] = Vortex.Blob(zblob[i],Γblob[i]+dΓ,δ)
#         press⁺_fd .= Plates.surface_pressure_inst(p⁺,motion,blobs⁺,(z₊,z₋),t,Δt,lesp,tesp,ss)
#         J[:,3i] .= (1.0/dΓ)*(press⁺_fd - press_fd)
#
#         blobs⁺[i] = blobs[i]
#
#     end
#
#     Plates.enforce_no_flow_through!(p⁺, motion, blobs, t)
#     press⁺_fd .= Plates.surface_pressure_inst(p⁺,motion,blobs,(z₊,z₋),t,Δt,lesp+dlesp,tesp,ss)
#     # dpdlesp_fd = (presslesp⁺_fd - press_fd)/dlesp
#     J[:,3*Nblob+1] .=  (1.0/dlesp)*(press⁺_fd - press_fd)
#     return J
# end
