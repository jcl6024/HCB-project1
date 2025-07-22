using LinearAlgebra

function chunks(L::Int,Nc::Int)
    """
    Given a parity symmetric Hamiltonian, this function determines the best way
    to partition such that each chunk has approximately the same number of
    elements.

    INPUTS:
    L: number of lattice sites
    Nc: number of chunks

    OUTPUT: 
    vector of Nc ranges corresponding to desired chunks.
    """
    MT = Int[]
    # calculate total number of elements
    F::Int = 0
    for i::Int in 1:L/2
        for j in range(i+1,L-(i-1))
            F+=1
        end
    end
    # calculate indices for which the total number of elements exceeds the threshhold
    # set by the total number of elements, i.e. F/Nc
    G::Int = 0
    for i::Int in 1:L/2
        for j in range(i+1,L-(i-1))
            G+=1
        end
        if G>F/Nc
            push!(MT,i)
            G=0
        end
    end
    push!(MT,L/2)
    ChunkVec = [] # abstract array--YIKES!
    for i::Int in 1:length(MT)
        if i>1
            push!(ChunkVec,(MT[i-1]+1):MT[i])
        else
            push!(ChunkVec,1:MT[i])
        end
    end
    return ChunkVec
end

chunks(1000,24)