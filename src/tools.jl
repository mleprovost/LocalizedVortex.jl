
export zero_small!, zero_small, bounds, findallfast, iqrfilter, removeoutliers, replaceoutliers!

function zero_small!(M::AbstractMatrix, tol)
    for ι in eachindex(M)
        if abs(M[ι]) ≤ tol
            M[ι] = 0
        end
    end
    M
end

zero_small(M::AbstractMatrix, tol) = zero_small!(copy(M), tol)

# Develop tools to remove the outliers based on the interquartile range method

function bounds(x::Array{Float64,1})

    q25, q75 = quantile(x, [0.25, 0.75])
    cutoff = 2.0*(q75 - q25)
    lower = q25 - cutoff
    upper = q75 + cutoff
    return lower, upper
end

# Faster implementation of findall from the thread
# https://discourse.julialang.org/t/findall-slow/30247/4
function findallfast(f, a::Array{T, N}) where {T, N}
    j = 1
    b = Vector{Int}(undef, length(a))
    @inbounds for i in eachindex(a)
        @inbounds if f(a[i])
            b[j] = i
            j += 1
        end
    end
    resize!(b, j-1)
    sizehint!(b, length(b))
    return b
end

function iqrfilter(X::Array{Float64,2})
    N, Ne = size(X)
    outliers = Int64[]
    @inbounds for i=1:N
        lower, upper = bounds(X[i,:])
        union!(outliers, findallfast(x-> x < lower || x > upper, X[i,:]))
    end
    return sort!(outliers)
end


function replaceoutliers!(X::Array{Float64,2}, outliers::Array{Int64,1})
    N, Ne = size(X)
    Nout = length(outliers)

    @inbounds for outlier in outliers
        # @show outlier
        idx = rand(1:Ne)
        # @show idx
        X[:,outlier] = deepcopy(X[:, idx])
    end
end

function removeoutliers(X::Array{Float64,2}, outliers::Array{Int64,1})
    N, Ne = size(X)
    Nout = length(outliers)

    Xout = zeros(N, Ne-Nout)

    count = 0
    @inbounds for i=1:Ne
        if !(i ∈ outliers)
            count += 1
            Xout[:,count] .= deepcopy(X[:, i])
        end
    end
    return Xout
end

#
# function replaceoutliers!(ens::EnsembleStateMeas{Nx, Ny, Ne}, outliers::Array{Int64,1}) where {Nx, Ny, Ne}
#     Nout = length(outliers)
#
#     @inbounds for outlier in outliers
#         # @show outlier
#         idx = rand(1:Ne)
#         # @show idx
#         ens.state.S[:,outlier] = deepcopy(ens.state.S[:, idx])
#         ens.meas.S[:,outlier] = deepcopy(ens.meas.S[:, idx])
#     end
# end
#
