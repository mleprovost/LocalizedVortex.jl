using Test


@testset "Test unitC" begin
    Nv = 10
    Γs = randn(Nv)
    zs = randn(Nv) + im * randn(Nv)

    δs = 1e-5*ones(Nv)
    blobs = map(Vortex.Blob, zs, Γs, δs)

    t = 1.0
    N = 512
    L = 1.0
    c = complex(1.0)
    ċ = 1.0
    α = π/9

    plate = Plate(N, L, c, α)
    motion = Plates.RigidBodyMotion(ċ, 0.0)

    z₊ = (blobs[end-1].z + 2plate.zs[end])/3
    z₋ = (blobs[end].z + 2plate.zs[1])/3
    znew = (z₊, z₋)
    Plates.enforce_no_flow_through!(plate, motion, blobs, t)

    Cᵛ = unitC(plate, blobs, t)
    @test norm(plate.C - Cᵛ*Γs)<1e-14
end

@testset "Test factstrength" begin

    Nv = 20
    Γs = randn(Nv)
    zs = randn(Nv) + im * randn(Nv)

    δs = 1e-5*ones(Nv)
    blobs = map(Vortex.Blob, zs, Γs, δs)

    t = 1.0
    N = 512
    L = 1.0
    c = complex(1.0)
    ċ = 1.0
    α = π/9

    plate = Plate(N, L, c, α)
    motion = Plates.RigidBodyMotion(ċ, 0.0)

    z₊ = (blobs[end-1].z + 2plate.zs[end])/3
    z₋ = (blobs[end].z + 2plate.zs[1])/3
    znew = (z₊, z₋)

    Plates.enforce_no_flow_through!(plate, motion, blobs, t)
    γ = copy(Plates.strength(plate))

    Cᵛ = unitC(plate, blobs, znew, t)
    aγ, bγ = factstrength(plate, motion, Cᵛ, Γs)

     #Infinite value at the edges
    @test norm(γ[2:end-1] - (aγ[:,1:end-2]*Γs + bγ)[2:end-1])<1e-12
end

@testset "Test factdifftangential" begin
    Nv = 20
    Γs = randn(Nv)
    zs = randn(Nv) + im * randn(Nv)

    δs = 1e-5*ones(Nv)
    blobs = map(Vortex.Blob, zs, Γs, δs)

    t = 1.0
    N = 512
    L = 1.0
    c = complex(1.0)
    ċ = 1.0
    α = π/9

    plate = Plate(N, L, c, α)
    motion = Plates.RigidBodyMotion(ċ, 0.0)

    z₊ = (blobs[end-1].z + 2plate.zs[end])/3
    z₋ = (blobs[end].z + 2plate.zs[1])/3
    znew = (z₊, z₋)

    Plates.enforce_no_flow_through!(plate, motion, blobs, t)

    vdiff = copy((Plates.Chebyshev.firstkind(real.(plate.C), plate.ss) .- Plates.tangent(ċ, α)))

    Cᵛ = unitC(plate, blobs, znew, t)
    adiff, bdiff = factdifftangential(plate, motion, Cᵛ)

    @test norm(vdiff - (adiff[:,1:end-2]*Γs + bdiff))<1e-13
end

@testset "Test factvorticity_flux" begin




end

@testset "Test factself_induce_velocity" begin
    N = 30
    Γs = randn(N)
    zs = randn(N) + im * randn(N)
    δs = 1e-5*ones(N)
    blobs = N > 0 ? map(Vortex.Blob, zs, Γs, δs) : Vortex.Blob[]
    t = 0.0

    srcvel = self_induce_velocity(blobs, t)

    K = factself_induce_velocity(blobs, t)

    @test norm(srcvel - K *Γs)<1e-14
end

@testset "Test factinduce_velocity" begin

    Nv = 30
    Γs = randn(Nv)
    zs = randn(Nv) + im * randn(Nv)
    δs = 1e-5*ones(Nv)
    blobs = map(Vortex.Blob, zs, Γs, δs)

    t = 1.0
    N = 512
    L = 1.0
    c = complex(t)
    ċ = complex(1.0)
    α = π/3

    plate = Plate(N, L, c, α)
    motion = Plates.RigidBodyMotion(ċ, 0.0)

    z₊ = (blobs[end-1].z + 2plate.zs[end])/3
    z₋ = (blobs[end].z + 2plate.zs[1])/3
    znew = (z₊, z₋)

    Plates.enforce_no_flow_through!(plate, motion, blobs, t)
    Cᵛ = unitC(plate, blobs, znew, t)

    srcvel = zeros(ComplexF64, length(blobs))
    induce_velocity!(srcvel, blobs, plate, t)

    ainduce, binduce = factinduce_velocity(blobs, plate, Cᵛ, t);

    @test norm(srcvel - (ainduce*Γs + binduce))<1e-10
end

@testset "Test factCdot" begin

    Nv = 20
    Γs = randn(Nv)
    zs = randn(Nv) + im * randn(Nv)
    δs = 1e-5*ones(Nv)
    blobs = map(Vortex.Blob, zs, Γs, δs)

    t = 1.0
    N = 256
    L = 1.0
    c = complex(t)
    ċ = complex(1.0)
    α = π/9

    plate = Plate(N, L, c, α)
    motion = Plates.RigidBodyMotion(ċ, 0.0)

    z₊ = (blobs[end-1].z + 2plate.zs[end])/3
    z₋ = (blobs[end].z + 2plate.zs[1])/3
    znew = (z₊, z₋)

    Plates.enforce_no_flow_through!(plate, motion, blobs, t)

    srcvel = self_induce_velocity(blobs, t)
    Plates.induce_velocity!(srcvel, blobs, plate, t)

    targvel = fill(ċ, length(plate))
    Ċt = zero(targvel)

    Plates.induce_acc!(Ċt, plate.zs, targvel, blobs, srcvel)
    n̂ = exp(-im*α)
    rmul!(Ċt, n̂)
    plate.dchebt! * Ċt

    # Computation from the compressed form
    plate = Plate(N, L, c, α)
    motion = Plates.RigidBodyMotion(ċ, 0.0)
    Plates.enforce_no_flow_through!(plate, motion, blobs, t)

    z₊ = (blobs[end-1].z + 2plate.zs[end])/3
    z₋ = (blobs[end].z + 2plate.zs[1])/3
    znew = (z₊, z₋)

    Cᵛ = unitC(plate, blobs, znew, t)
    K = factself_induce_velocity(blobs, t)
    ainduce, binduce = factinduce_velocity(blobs, plate, Cᵛ, t);

    aĊ, bĊ = factCdot(plate, motion, blobs, Cᵛ, t)

    Ċ = zeros(ComplexF64, N)

    @tensor Ċ[a] = aĊ[a,b,c]*Γs[b]*Γs[c] + bĊ[a, d]*Γs[d]


    @test norm(Ċ - Ċt)<1e-10
end
