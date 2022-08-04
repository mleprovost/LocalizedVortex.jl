
export CacheJacobian, Jacobian!, Jacobian,
       CacheHessian, Hessian!, Hessian

# This function computes the Jacobian for h: R^n → R^m using central differencing


struct CacheJacobian
    Ny::Int64
    Nx::Int64
    xp::Array{Float64, 1}
    xm::Array{Float64, 1}
    hx::Array{Float64, 1}
    ϵ::Float64
end

CacheJacobian(Ny, Nx; fdtype::Symbol=:central) = CacheJacobian(Ny, Nx, zeros(Nx), zeros(Nx), zeros(Ny),
                                                               FiniteDiff.default_relstep(Val{fdtype}, eltype(1.0)))

# This version uses
# central differencing with fdtype = :central, i.e. ∂_{i} h = (h(x + ϵ ei) - h(x - ϵ ei))/2ϵ, or
# forward differencing with fdtype = :forward, i.e. ∂_{i} h = (h(x + ϵ ei) - h(x))/ϵ
# to compute the Jacobian

function Jacobian!(J, h::Function, x, cache::CacheJacobian; fdtype::Symbol=:central)
    @assert size(J) == (cache.Ny, cache.Nx)
    @assert size(x,1) == cache.Nx
    ϵ = cache.ϵ
    copy!(cache.xp, x)
    if fdtype == :central
        copy!(cache.xm, x)
        @inbounds for i = 1:cache.Nx
            x_old = x[i]
            cache.xp[i] += ϵ
            cache.xm[i] -= ϵ
            view(J,:,i) .= h(cache.xp) - h(cache.xm)
            cache.xp[i] = x_old
            cache.xm[i] = x_old
        end
        rmul!(J, 0.5/ϵ)
    elseif fdtype == :forward
        cache.hx .= h(x)
        @inbounds for i = 1:cache.Nx
            x_old = x[i]
            cache.xp[i] += ϵ
            view(J,:,i) .= h(cache.xp) - cache.hx
            cache.xp[i] = x_old
        end
        rmul!(J, 1/ϵ)
    end
    return J
end

Jacobian(h::Function, x, Ny, Nx; fdtype::Symbol=:central) = Jacobian!(zeros(Ny, Nx), h, x, CacheJacobian(Ny, Nx; fdtype = fdtype); fdtype = fdtype)
Jacobian(h::Function, x, cache::CacheJacobian; fdtype::Symbol=:central) = Jacobian!(zeros(cache.Ny, cache.Nx), h, x, cache; fdtype = fdtype)

struct CacheHessian
    Ny::Int64
    Nx::Int64
    hx::Array{Float64,1}
    xpp::Array{Float64, 1}
    xpm::Array{Float64, 1}
    xmp::Array{Float64, 1}
    xmm::Array{Float64, 1}
    ϵ::Float64
end

CacheHessian(Ny, Nx) = CacheHessian(Ny, Nx, zeros(Ny), zeros(Nx), zeros(Nx), zeros(Nx), zeros(Nx),
                        FiniteDiff.default_relstep(Val{:central}, eltype(1.0)))

# This version uses ∂^2_{i,i} h = (h(x + ϵ ei) - 2h(x) + h(x - ϵ ei))/ϵ^2 for the diagonal entries
# and ∂^2_{i,j} = (h(x + ϵ ei + ϵ ej) - h(x + ϵ ei - ϵ ej) - h(x - ϵ ei + ϵ ej) + h(x - ϵ ei - ϵ ej))/(4ϵ^2)

function Hessian!(H, h::Function, x, cache::CacheHessian)
    @assert size(H) == (cache.Ny, cache.Nx, cache.Nx)
    @assert size(x,1) == cache.Nx
    ϵ = cache.ϵ
    copy!(cache.xpp, x)
    cache.hx .= h(x)
    copy!(cache.xmm, x)
    # Fill diagonal entries
    @inbounds for i = 1 : cache.Nx
        cache.xpp[i] += ϵ
        cache.xmm[i] -= ϵ
        view(H,:,i,i) .= (1.0/ϵ^2)*(h(cache.xpp) - 2.0*cache.hx +  h(cache.xmm))
        cache.xpp[i] -= ϵ
        cache.xmm[i] += ϵ
    end

    # Fill off-diagonal entries and use symmetry
    copy!(cache.xpm, x)
    copy!(cache.xmp, x)

    for i = 1:cache.Nx
        cache.xpp[i] += ϵ
        cache.xpm[i] += ϵ
        cache.xmp[i] -= ϵ
        cache.xmm[i] -= ϵ

        for j = i+1:cache.Nx
            cache.xpp[j] += ϵ
            cache.xpm[j] -= ϵ
            cache.xmp[j] += ϵ
            cache.xmm[j] -= ϵ

            view(H,:,i,j) .= (0.25/ϵ^2)*(h(cache.xpp) - h(cache.xpm) - h(cache.xmp) + h(cache.xmm))
            H[:,j,i] .= view(H,:,i,j)
            cache.xpp[j] -= ϵ
            cache.xpm[j] += ϵ
            cache.xmp[j] -= ϵ
            cache.xmm[j] += ϵ
        end
        cache.xpp[i] -= ϵ
        cache.xpm[i] -= ϵ
        cache.xmp[i] += ϵ
        cache.xmm[i] += ϵ
    end

    # @show norm(cache.xpp-x)
    # @show norm(cache.xpm-x)
    # @show norm(cache.xmp-x)
    # @show norm(cache.xmm-x)
    return H
end

Hessian(h::Function, x, Ny, Nx) = Hessian!(zeros(Ny, Nx, Nx), h, x, CacheHessian(Ny, Nx))
Hessian(h::Function, x, cache::CacheHessian) = Hessian!(zeros(cache.Ny, cache.Nx, cache.Nx), h, x, cache)


# Compute simultaneously the Hessian and Jacobian
struct CacheJacHessian
    Ny::Int64
    Nx::Int64
    hx::Array{Float64,1}
    hxpi::Array{Float64,1}
    hxpj::Array{Float64,1}
    hxmi::Array{Float64,1}
    hxmj::Array{Float64,1}
    xpij::Array{Float64, 1}
    xpi::Array{Float64, 1}
    xpj::Array{Float64, 1}
    xmi::Array{Float64, 1}
    xmj::Array{Float64, 1}
    xmij::Array{Float64, 1}
    ϵ::Float64
end

CacheJacHessian(Ny, Nx) = CacheJacHessian(Ny, Nx, zeros(Ny), zeros(Ny), zeros(Ny), zeros(Ny), zeros(Ny), zeros(Nx), zeros(Nx), zeros(Nx), zeros(Nx),
                        FiniteDiff.default_relstep(Val{:central}, eltype(1.0)))

# This version computes Jacobian and Hessian at the same to reuse computations
# ∂_{i} h = (h(x + ϵ ei) - h(x - ϵ ei))/2ϵ to compute the Jacobian
# ∂^2_{i,i} h = (h(x + ϵ ei) - 2h(x) + h(x - ϵ ei))/ϵ^2 for the diagonal entries
# and ∂^2_{i,j} = (h(x + ϵ ei + ϵ ej) - h(x + ϵ ei ) - h(x + ϵ ej) + 2 h(x) - h(x - ϵ ei) -h(x - ϵ ej) + h(x - ϵ ei - ϵ ej))/(2ϵ^2)
# See https://en.wikipedia.org/wiki/Finite_difference
function JacHessian!(J, H, h::Function, x, cache::CacheJacHessian)
    @assert size(J) == (cache.Ny, cache.Nx)
    @assert size(H) == (cache.Ny, cache.Nx, cache.Nx)
    @assert size(x,1) == cache.Nx
    ϵ = cache.ϵ
    copy!(cache.xpp, x)
    cache.hx .= h(x)
    copy!(cache.xmm, x)
    # Fill diagonal entries
    @inbounds for i = 1 : cache.Nx
        cache.xpp[i] += ϵ
        cache.xmm[i] -= ϵ
        view(H,:,i,i) .= (1.0/ϵ^2)*(h(cache.xpp) - 2.0*hx +  h(cache.xmm))
        cache.xpp[i] -= ϵ
        cache.xmm[i] += ϵ
    end

    # Fill off-diagonal entries and use symmetry
    copy!(cache.xpm, x)
    copy!(cache.xmp, x)

    @showprogress for i = 1:cache.Nx
        cache.xpij += ϵ
        cache.xmij -= ϵ

        for j = i+1:cache.Nx
                cache.xpij += ϵ
                cache.xmij -= ϵ

                # view(H, :, i, j) .=


        end
    end

    # @show norm(cache.xpp-x)
    # @show norm(cache.xpm-x)
    # @show norm(cache.xmp-x)
    # @show norm(cache.xmm-x)
    return H
end

JacHessian(h::Function, x, Ny, Nx) = JacHessian!(zeros(Ny, Nx), zeros(Ny, Nx, Nx), h, x, CacheHessian(Ny, Nx))
JacHessian(h::Function, x, cache::CacheJacHessian) = JacHessian!(zeros(cache.Ny, cache.Nx), zeros(cache.Ny, cache.Nx, cache.Nx), h, x, cache)

# function sparsity_offline(h::Function, X, t, Ny, Nx, ϵy::AdditiveInflation, δ::Float64)
#
#
#     Ne, NxX = size(X)
#     @assert NxX == Nx "size of the ensemble is not consistent"
#     Nypx = Ny + Nx
#     Ω = zeros(Nypx, Nypx)
#
#     Jx = zeros(Ny, Nx)
#     Hx = zeros(Ny, Nx, Nx)
#
#     @inbounds for i=1:Ne
#
#
#
#     end
#
#
#
#
#
#
# end
