


@testset "Test diff_firstkind" begin

    config = let N = 128, L = 1.0, ċ = 1.0,
                 α = π/9, δ = 9e-3, Δt = 1e-2,
                 tesp = 0.0,
                 ϵX = 1e-5, ϵΓ = 1e-5, ϵLESP = 8.5e-5,
                 β = 1.028,
                 ϵY = 1e-4

        VortexParams(N, L, ċ, α, δ, Δt, true, tesp, ϵX, ϵΓ, ϵLESP, β, ϵY)
    end

    t = 1.0

    plate = Plate(config.N, config.L, complex(t + config.Δt), config.α)

    # C = zeros(ComplexF64, p.N);
    C = randn(p.N) + im*randn(p.N);

    B = real.(C)

    JBexact = diff_firstkind(plate, plate.ss[1:50])
    JB_fd = Jacobian(x->PotentialFlow.Plates.Chebyshev.firstkind(x, plate.ss[1:50]), B, length(plate.ss[1:50]), plate.N)


    @test norm(JBeaxct-JB_fd)<1e-8

     #T0(s) = 1 and T1(s) = s
    @test norm(JBexact[:,1] - ones(length(plate.ss[1:50])))<1e-8
    @test norm(JBexact[:,2] - plate.ss[1:50])<1e-8

end

@testset "Test derivative of Point kernel" begin
    psource = Vortex.Point(randn() + im*randn(), randn())

    source = [real(psource.z); imag(psource.z); psource.S]

    ztarget = randn() + im*randn()

    G_exact = zeros(ComplexF64, 3)
    diff_induce_velocity!(G_exact, ztarget, psource, 0.0)

    # Check with finite differences
    cache = CacheJacobian(1, 3)

    G_fd = zeros(ComplexF64, 1, 3)

    Jacobian!(G_fd, x->induce_velocity(ztarget, Vortex.Point(x[1] + im*x[2], x[3]), 0.0), source, cache)

    @test norm(G_exact - G_fd[1,:])<1e-8
end

@testset "Test derivative of Blob kernel" begin
    for i=1:10
        δ = 1e-3
        bsource = Vortex.Blob(randn() + im*randn(), randn(), δ)

        source = [real(bsource.z); imag(bsource.z); bsource.S]

        ztarget = randn() + im*randn()

        G_exact = zeros(ComplexF64, 3)
        diff_induce_velocity!(G_exact, ztarget, bsource, 0.0)

        # Check with finite differences
        cache = CacheJacobian(1, 3)

        G_fd = zeros(ComplexF64, 1, 3)

        Jacobian!(G_fd, x->induce_velocity(ztarget, Vortex.Blob(x[1] + im*x[2], x[3], δ), 0.0), source, cache)

        @test norm(G_exact - G_fd[1,:])<1e-4
    end
end

@testset "Test derivative of enforce_no_flow_through!" begin
    config = let N = 128, L = 1.0, ċ = 1.0,
                 α = π/9, δ = 9e-3, Δt = 1e-2,
                 tesp = 0.0,
                 ϵX = 1e-5, ϵΓ = 1e-5, ϵLESP = 8.5e-5,
                 β = 1.028,
                 ϵY = 1e-4

        VortexParams(N, L, ċ, α, δ, Δt, true, tesp, ϵX, ϵΓ, ϵLESP, β, ϵY)
    end

    Nv = 60
    xtest = randn(3*Nv)
    t = 1.0

    function test(x)
        x = vcat(x, 0.5, 0.0)

        plate = Plate(config.N, config.L, complex(t + config.Δt), config.α)
        motion = Plates.RigidBodyMotion(config.ċ, 0.0)

        blobs, lesp, tesp = state_to_blobs(x, config.δ);

        Plates.enforce_no_flow_through!(plate, motion, blobs, t)
        return plate.C
    end

    # Computation to test
    plate = Plate(config.N, config.L, complex(t + config.Δt), config.α)
    motion = Plates.RigidBodyMotion(config.ċ, 0.0)
    JC_exact = zeros(ComplexF64, plate.N, 3*Nv);
    xtest_aug = vcat(xtest, 0.5, 0.0)


    blobs, lesp, tesp = state_to_blobs(xtest_aug, config.δ)
    diff_enforce_no_flow_through!(JC_exact, plate, motion, blobs, t)

    # Computation FD
    cache = CacheJacobian(plate.N, 3*length(blobs))
    JC_fd = zeros(ComplexF64, plate.N, 3*length(blobs))

    Jacobian!(JC_fd, x->test(x), xtest, cache)

    @test norm(JC_exact - JC_fd)<1e-5
end
