using Test

@testset "Test orthogonal projector Nx = 1" begin
    v = [1.0]

    P = OrthProjector(v)

    @test P.ortho == true
    @test P.Nx == 1
    @test P.r == 1
    @test P.P == reshape([1.0],(1,1))

    @test P.P*P.P == P.P
    @test P.P == P.P'
    @test P.P == P.v*P.v'

    x = randn(1)
    xprime = zero(x)
    xperp  = zero(x)

    project!(P, xprime, x)

    @test xprime == x
    @test xprime == P.v*P.v'*x

    xprime = zero(x)
    xperp  = zero(x)

    decompose!(P, xprime, xperp, x)

    @test xprime == x
    @test xperp == [0.0]
    @test xperp == (I-P.P)*xprime
end

@testset "Test orthogonal projector Nx>1 with non-orthognal family" begin
    v = randn(1)
    P = OrthProjector(v)

    @test P.ortho == false
    @test P.Nx == 1
    @test P.r == 1
    @test P.P == reshape([1.0], (1,1))

    @test P.P*P.P == P.P

    x = randn(1)
    xprime = zero(x)
    xperp  = zero(x)

    project!(P, xprime, x)

    @test xprime == (1/dot(v,v))*(v*v')*x

    xprime = zero(x)
    xperp  = zero(x)
    decompose!(P, xprime, xperp, x)

    @test xprime == x
    @test xperp == [0.0]
end


@testset "Test orthogonal projector Nx>1" begin
    Nx = 10
    r = 5
    u = randn(Nx, r)

    F = qr(u)
    Q = Matrix(F.Q)
    R = F.R

    @test norm(Q*R - u) < 1e-14

    @test abs(norm(Q[:,1]) - 1.0)<1e-14
    @test abs(dot(Q[:,1], Q[:,2])) < 1e-14

    P = OrthProjector(Q)

    @test P.ortho == true
    @test P.Nx == 10
    @test P.r == 5
    @test norm(P.P - Q*Q')<1e-14

    @test norm(P.P*P.P - P.P)<1e-14
    @test norm(P.P - P.P')<1e-14

    x = randn(Nx)

    xprime = zero(x)


    project!(P, xprime, x)
    @test norm(xprime - P.v*P.v'*x)<1e-14

    x = randn(Nx)
    xprime = zero(x)
    xperp  = zero(x)

    decompose!(P, xprime, xperp, x)

    @test norm(x - (xprime + xperp))<1e-14
    @test abs(dot(xprime, xperp))<1e-14
end

@testset "Test orthogonal projector Nx>1 with non-orthognal family" begin
    Nx = 3
    r = 2
    # This is a non-orthogonal basis for a subspace of dimension 2 of R^3
    u = [1.0 1.0; 2.0 -1.0; 3.0 1.0]


    P = OrthProjector(u)

    @test P.ortho == false
    @test P.Nx == Nx
    @test P.r == r
    @test norm(P.P - u*(inv(u'*u)*u'))<1e-14

    @test norm(P.P*P.P - P.P)<1e-14
    @test norm(P.P - P.P')<1e-14

    x = randn(Nx)

    xprime = zero(x)


    project!(P, xprime, x)
    @test norm(xprime - P.v*inv(P.v'*P.v)*P.v'*x)<1e-14

    x = randn(Nx)
    xprime = zero(x)
    xperp  = zero(x)

    decompose!(P, xprime, xperp, x)

    @test norm(x - (xprime + xperp))<1e-14
    @test abs(dot(xprime, xperp))<1e-14
end
