
using PotentialFlow: Points, Plates

export jacobian_pressure!, jacobian_pressure

# In-place version
function jacobian_pressure!(Jac, state, t, config, ss; Nplate::Int64 = 64)
    blobs, lesp, tesp = state_to_blobs(state, config.δ)
    motion = Plates.RigidBodyMotion(config.ċ, 0.0)
    zs = complex(t + config.Δt) .+ 0.5config.L*[-1.0; 1.0]*exp(im*config.α)
    z₊ = (blobs[end-1].z + 2zs[end])/3
    z₋ = (blobs[end].z + 2zs[1])/3

    function compute_pressure(v)
        ptmp = PotentialFlow.Plate{Elements.property_type(eltype(v))}(Nplate, config.L, complex(t + config.Δt), config.α)
        return complex(Plates.surface_pressure_inst(ptmp,motion,v,(z₊,z₋),t,config.Δt,lesp,tesp, ss))
    end

    dpdz, _ = PotentialFlow.Elements.jacobian_position(compute_pressure,blobs)

    Jac[:,1:3:end-1] .=  2.0*real.(dpdz)
    Jac[:,2:3:end-1] .= -2.0*imag.(dpdz)
    Jac[:,3:3:end-1] .=  real.(PotentialFlow.Elements.jacobian_strength(compute_pressure,blobs))

    function lesp_to_pressure(v,lesp)
        ptmp = PotentialFlow.Plate{Elements.property_type(eltype(v))}(Nplate, config.L, complex(t + config.Δt), config.α)
        press = Plates.surface_pressure_inst(ptmp,motion,v,(z₊,z₋),t,config.Δt,lesp,tesp, ss)
        return complex(press)
    end
    # In the end, we want to use only function and the associated anonymous function
    Jac[:,end:end] .= PotentialFlow.Elements.jacobian_param(lesp_to_pressure,(blobs,lesp))

    return Jac
end

jacobian_pressure(state, t, config, ss; Nplate::Int64 = 64) = jacobian_pressure!(zeros(size(ss,1), size(state,1)-1), state, t, config, ss; Nplate = Nplate)
