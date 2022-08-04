export setup_vortexorder


# Nx size of the state
# off_p contains the order for the off-diagonal component
# off_rad contains the radius for the localization of the map
# nonid_rad contains the number of non id components
# dist matrix of the distances between the variables
function setup_vortexorder(diagobs_p, diagunobs_p, off_p, off_rad, nonid_rad, dxx::Array{Float64,2}, dxy::Array{Float64,2})
    Nx, Ny = size(dxy)

    @assert size(dxx,1)==size(dxx,2)
    @assert size(dxx,1)==size(dxy,1)
    # perm = sortperm(view(dist,:,1))
    # perm = sortperm(mean(dxy, dims = 2)[:,1])
    perm = sortperm(view(dxy,:,Ny))

    dist_to_order = fill(-1, Nx)
    # fill!(view(dist_to_order,1:ceil(Int64,off_rad)), off_p)
    fill!(view(dist_to_order,1:ceil(Int64,min(off_rad, Nx))), off_p)
    order = [fill(-1,i) for i=1:Nx+1]
    nonid = Int64[]

    for i=1:Nx
        node_i = perm[i]
        # dist_i1 = dist[node_i,1]
        dist_i1 = dxy[node_i,end]
        # Vector to store order, offset of 1
        # because we have one scalar observation
        order_i = zeros(Int64, i-1)#view(order,i+1)
        if dist_i1 <= nonid_rad
            if i>1
            for j=1:i-1
                # extract distance to node i
                node_j = perm[j]
                dist_ij = dxx[node_i, node_j]
                # compute order for variable based on distance
                if dist_ij <= off_rad
                    order_i[j] = off_p
                else
                    order_i[j]= -1
                end
            end
            end
            # Define order for diagonal TM components and observation component
            if i==1 # observed mode
                order[2][2] = diagobs_p
                order[2][1] = off_p
            else # unobserved mode
                order[i+1][2:i] .= deepcopy(order_i)
                order[i+1][i+1] = diagunobs_p
                # order[i+1][1] = -1
                order[i+1][1] = off_p
            end
        end
    end
    return order
end
