

using Test
@testset "Test the distance matrix state/measurements" begin

    config = let N = 10, L = 1.0, ċ = 1.0,
                 α = 0.0, δ = 9e-3, Δt = 1e-2,
                 tesp = 0.0,
                 ϵX = 1e-5, ϵΓ = 1e-5, ϵLESP = 8.5e-5,
                 β = 1.028,
                 ϵY = 1e-4

        VortexParams(N, L, ċ, α, δ, Δt, true, tesp, ϵX, ϵΓ, ϵLESP, β, ϵY)
    end
    idxp = [1;2;4;config.N]
    c = complex(0.0)
    state = [-0.5;1.0; 0.5; 0.5; 1.0; -0.5; 0.2; 0.3; 0.4; 0.5]

    d = dstatemeas(state, c, config, idxp)

    @test size(d)==(10,4)

    @test d[1:3,1] == ones(3)

    @test d[4:6,4] == ones(3)

    @test d[end-3:end-1,4] == abs(0.2+im*0.3 - (c+ config.L/2*exp(im*config.α)))*ones(3)

    zLE = c + 0.5config.L*exp(im*config.α)


    @test d[end,1]   == exp(3*abs(zLE-(c - config.L/2*exp(im*config.α))))-1
    @test d[end,end] == exp(3*abs(zLE-(c + config.L/2*exp(im*config.α))))-1

end
