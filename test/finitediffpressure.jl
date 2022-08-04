
@testset "test finite difference pressure" begin

    DELTA=1e-6
    BIGEPS = 10000*eps(1.0)
    TOL=5e-6
    BIGTOL=5e-4
    BIGGESTTOL=1e-3

    safenorm(a) = norm(filter(x -> ~isnan(x),a))

    nblob = 20
    pos = rand(ComplexF64,nblob)
    str = rand(length(pos))

    σ = 1e-2
    t = 0.0
    N = 128
    L = 2.0
    Δt = 0.01
    c = complex(0.0)
    α = 0.0
    ċ = complex(1.0)
    α̇ = 0.0
    lesp = 0.2
    tesp = 0.0

    blobs = Vortex.Blob.(pos,str,σ)
    config = VortexParams(N, L, ċ, α, σ, Δt, true, tesp, 0.0, 0.0, 0.0, 0.0, 0.0)

    J = finitediff_pressure(blobs, complex(0.0), 0.0, Δt, lesp, tesp, config)

    for i=1:nblob
        # For finite difference approximations
        dz = zeros(ComplexF64,length(blobs))
        dz[i] = DELTA
        dΓ = zeros(Float64,length(blobs))
        dΓ[i] = DELTA
        blobsx⁺ = Vortex.Blob.(Elements.position(blobs).+dz,Elements.circulation.(blobs),σ)
        blobsy⁺ = Vortex.Blob.(Elements.position(blobs).+im*dz,Elements.circulation.(blobs),σ)
        blobsΓ⁺ = Vortex.Blob.(Elements.position(blobs),Elements.circulation.(blobs).+dΓ,σ)

        p = PotentialFlow.Plate(N,L,c,α)
        motion = PotentialFlow.RigidBodyMotion(ċ,α̇ )

        Plates.enforce_no_flow_through!(p, motion, blobs, 0.0)
        px⁺ = deepcopy(p)
        Plates.enforce_no_flow_through!(px⁺, motion, blobsx⁺, 0.0)
        py⁺ = deepcopy(p)
        Plates.enforce_no_flow_through!(py⁺, motion, blobsy⁺, 0.0)
        pΓ⁺ = deepcopy(p)
        Plates.enforce_no_flow_through!(pΓ⁺, motion, blobsΓ⁺, 0.0)


        z₊ = (blobs[end-1].z + 2p.zs[end])/3
        z₋ = (blobs[end].z + 2p.zs[1])/3

        plesp⁺ = deepcopy(p)
        dlesp = DELTA

        press_fd = Plates.surface_pressure_inst(p,motion,blobs,(z₊,z₋),0.0,Δt,lesp,tesp)
        pressx⁺_fd = Plates.surface_pressure_inst(px⁺,motion,blobsx⁺,(z₊,z₋),0.0,Δt,lesp,tesp)
        pressy⁺_fd = Plates.surface_pressure_inst(py⁺,motion,blobsy⁺,(z₊,z₋),0.0,Δt,lesp,tesp)
        pressΓ⁺_fd = Plates.surface_pressure_inst(pΓ⁺,motion,blobsΓ⁺,(z₊,z₋),0.0,Δt,lesp,tesp)

        dpdx_fd = (pressx⁺_fd - press_fd)/dz[i]
        dpdy_fd = (pressy⁺_fd - press_fd)/dz[i]
        dpdz_fd = 0.5*(dpdx_fd - im*dpdy_fd)
        dpdzstar_fd = 0.5*(dpdx_fd + im*dpdy_fd)
        dpdΓ_fd = (pressΓ⁺_fd - press_fd)/dΓ[i]

        function compute_pressure(v)
          ptmp = PotentialFlow.Plate{Elements.property_type(eltype(v))}(N,L,c,α)
          return complex(Plates.surface_pressure_inst(ptmp,motion,v,(z₊,z₋),0.0,Δt,lesp,tesp))
        end

        dpdz, dpdzstar = jacobian_position(compute_pressure,blobs)

        @test isapprox(safenorm(dpdz[:,i]-dpdz_fd)/safenorm(dpdz_fd),0.0,atol=BIGTOL)
        @test isapprox(safenorm(dpdzstar[:,i]-dpdzstar_fd)/safenorm(dpdzstar_fd),0.0,atol=BIGTOL)

        dpdΓ = real(jacobian_strength(compute_pressure,blobs))

        @test isapprox(safenorm(dpdΓ[:,i]-dpdΓ_fd)/safenorm(dpdΓ_fd),0.0,atol=BIGTOL)

        presslesp⁺_fd = Plates.surface_pressure_inst(plesp⁺,motion,blobs,(z₊,z₋),0.0,Δt,lesp+dlesp,tesp)
        dpdlesp_fd = (presslesp⁺_fd - press_fd)/dlesp

        function lesp_to_pressure(v,lesp)
          ptmp = PotentialFlow.Plate{Elements.property_type(eltype(v))}(N,L,c,α)
          press = Plates.surface_pressure_inst(ptmp,motion,v,(z₊,z₋),0.0,Δt,lesp,tesp)
          return complex(press)
        end

        dpdlesp = jacobian_param(lesp_to_pressure,(blobs,lesp))

        @test isapprox(safenorm(dpdlesp-dpdlesp_fd)/safenorm(dpdlesp),0.0,atol=BIGTOL)
        @test isapprox(safenorm(J[:,3*i-2]-real.(dpdx_fd))/safenorm(J[:,3*i-2]), 0.0, atol=1e-10)
        @test isapprox(safenorm(J[:,3*i-1]-real.(dpdy_fd))/safenorm(J[:,3*i-1]), 0.0, atol=1e-10)
        @test isapprox(safenorm(J[:,3*i]-real.(dpdΓ_fd))/safenorm(J[:,3*i]), 0.0, atol=1e-10)
        @test isapprox(safenorm(J[:,end]-real.(dpdlesp_fd))/safenorm(J[:,end]), 0.0, atol=1e-10)

    end
end
