export compute_sourceẋ!, pressure_source!, pressure_source

function compute_sourceẋ!(ẋ, x, freestream, t)
    # Zero the velocity
    reset_velocity!(ẋ)

    # Compute the self-induced velocity of the system
    self_induce_velocity!(ẋ, x, t)

    induce_velocity!(ẋ, x, freestream, t)

    # Overwrite the velocity of the mirrored sources
    fill!(ẋ[end-1], zero(Complex{Elements.property_type(eltype(x[end-1]))}))
    fill!(ẋ[end], zero(Complex{Elements.property_type(eltype(x[end]))}))
end

function pressure_source!(press, targetvels, targetϕ, sourcevels, target, source, freestream, t, Δt)
    source = deepcopy(source)

    reset_velocity!(sourcevels)
    reset_velocity!(targetvels)

    # Compute the self-velocity
    compute_ẋ!(sourcevels, source, freestream, t)

    # Compute the induced velocity on the target elements
    induce_velocity!(targetvels, target, (source, freestream), t)

    targetϕ .= real.(complexpotential(target, (source, freestream)))

    # advective term
    fill!(press, 0.0)
    press .= -0.5*abs2.(targetvels)

    # unsteady term
    advect!(source, source, sourcevels, Δt)
    press .+= (targetϕ - real.(complexpotential(target, (source, freestream))))/Δt

    return press
end

pressure_source(target, source, freestream, t, Δt) = pressure_source!(zeros(Float64, length(target)),
                                                        allocate_velocity(target),
                                                        zeros(Float64, length(target)),
                                                        allocate_velocity(source),
                                                        target, source, freestream, t, Δt)
