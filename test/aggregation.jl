
@testset "Test tools for aggregation procedure" begin
      nblob = 30
      pos = 0.1*im .+ randn(nblob) + 1.0im*rand(nblob)
      str = 1.0*randn(length(pos))

      δ = 9e-3
      blobs = Vortex.Blob.(pos,str,δ)

      t = 0.0
      Δt = 0.01

      Γ = circulation.(blobs)

      plate = Plate(512, 1.5, complex(t), π/12)
      motion = Plates.RigidBodyMotion(1.0, 0.0)

      Plates.enforce_no_flow_through!(plate, motion, blobs, t)

      @test norm(matrix_self_induce_velocity_blobs(blobs,t)*Γ-self_induce_velocity(blobs, t))<1e-14
      @test norm(matrix_induce_velocity_blobs(plate.zs, blobs,t)*Γ-induce_velocity(plate.zs, blobs, t))<1e-14
      @test norm(matrix_induce_velocity_singular(plate.zs, blobs, t)*Γ-induce_velocity(plate, blobs, t))<1e-14
      @test norm((Plates.strength(plate)-strength_motion(plate)-real(matrix_strength_vortex(plate, blobs, t)*Γ))[2:end-1])<5e-13

      @test norm(matrix_suction_parameters(plate, blobs)*Γ .+ 4.0* [-0.5; 0.5]*imag(exp(-im*plate.α)*motion.ċ) -
      [Plates.suction_parameters(plate)[1]; Plates.suction_parameters(plate)[2]])<1e-14

      A, b = linearsystem_aggregation(plate, blobs, t, plate.ss[2:end-1])
      # Verify that Ax = b
      @test norm(A*Γ-b)<1e-13

      A, b = linearsystem_aggregation(plate, blobs, t, plate.ss)
      # Verify self-induce-velocity
      norm((A[1:size(blobs,1),:] + im*A[1+size(blobs,1):2*size(blobs,1),:])*Γ-self_induce_velocity(blobs,t))<1e-14

      # Verify bound vortex sheet strength
      @test norm((Plates.strength(plate)-strength_motion(plate)-A[2*size(blobs,1)+2*size(plate.ss,1)+1:2*size(blobs,1)+3*size(plate.ss,1),:]*Γ)[2:end-1])<5e-13

      # Verify edge conditions

      Plates.enforce_no_flow_through!(plate, motion, blobs, t)
      @test norm(A[end-2:end-1,:]*Γ .+ 4.0* [-0.5; 0.5]*imag(exp(-im*plate.α)*motion.ċ) -
            [Plates.suction_parameters(plate)[1]; Plates.suction_parameters(plate)[2]])<1e-13

      # Verify total circulation
      @test norm(A[end:end,:]*Γ .- sum(Γ))<1e-13

      # Verify induced velocity on a set of target locations
      Areal = linearsystem_aggregation(plate, blobs, t, plate.ss)[1][2*nblob+1:2*nblob+size(plate.ss,1),:]
      Aimag = linearsystem_aggregation(plate, blobs, t, plate.ss)[1][2*nblob+size(plate.ss,1)+1:2*nblob+2*size(plate.ss,1),:]
      # norm((matrix_aggregation(plate, blobs, t, plate.ss)[2*size(blobs,1)+1:2*size(blobs,1)+size(plate.ss,1),:] + im*(matrix_aggregation(plate, blobs, t, plate.ss)[2*size(blobs,1)+1:2*size(blobs,1)+size(plate.ss,1),:]))*Γ-induce_velocity(plate, blobs, t)*exp(+im*plate.α))
      @test norm((Areal+im*Aimag)*Γ-induce_velocity(plate, blobs, t)*(0.5*exp(-im*plate.α)*plate.L))<1e-13
end
