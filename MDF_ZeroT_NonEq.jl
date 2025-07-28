"""
24/07/25 V1

Zero Temperature Out-of-Equilibrium Momentum Distribution Function for Hard Core Bosons.
"""


#########################
### Import Statements ###
#########################
using LinearAlgebra
using Plots
using NLsolve
using Optim
using NonlinearSolve 
using Roots
using Distributed
using LinearAlgebra
###############
### Include ###
###############
include("Hamiltonian_functions.jl")
include("SF_functions.jl")
include("ChunksP.jl") # P for parity

###############################
### One-body Density Matrix ###
###############################
function O(j::Int,L::Int)
    """
    Return a diagonal matrix wherein the first j-1 entries are -1 and all
    others 1.

    INPUTS
    j: site index
    L: number of lattice sites

    OUTPUT
    LxL diagonal matrix.
    """
    Id::Matrix{Int} = Matrix(1I,L,L)
    for i in range(1,j-1)
        Id[i,i]=-1
    end
    return Id
end

function Pjt(t::Float64,j::Int64,L::Int64,N::Int64,U::Matrix{Float64},U2::Matrix{Float64},E2::Vector{Float64})
    """
    Construct the matrix P of coefficients of an initial state (assuming
    ground state) with additional row corresponding to creation of a particle
    at the site j and corresponding signs owing to Jordan-Wigner strings.

    INPUTS
    t: time variable
    j: site index
    L: number of lattice sites
    N: number of particles
    U: unitary matrix of components of single particle eigenstates pre-quench
    U2: unitary matrix of components of single particle eigenstates post-quench
    E2: vector of post-quench eigenenergies

    OUTPUTS
    Lx(N+1) matrix of components of single-particle eigenstates after action
    of Jordan-Wigner strings and particle creation at site j.
    """
    colj::Vector{Int} = zeros(Int,L)
    colj[j] = 1
    Dt::Matrix{ComplexF64} = Diagonal(exp.(complex(0,-t)*E2))
    Pt::Matrix{ComplexF64} = U2 * Dt * adjoint(U2) * U[:,1:N]
    P1::Matrix{ComplexF64} = O(j,L)*([Pt;;colj])
    return P1
end

function Gijt(i::Int64,j::Int64,t::Float64,L::Int64,N::Int64,U::Matrix{Float64},U2::Matrix{Float64},E2::Vector{Float64})
    """
    Calculate one-body Green's function for i!=j.

    INPUTS
    i,j: site index
    t: time
    L: number of lattice sites
    N: number of particles
    U: unitary matrix of components of single particle eigenstates pre-quench
    U2: unitary matrix of components of single particle eigenstates post-quench
    E2: vector of post-quench eigenenergies

    OUTPUT
    ji entry of the correlation matrix.
    """
    matprod::Matrix{ComplexF64} = transpose(conj(Pjt(t,j,L,N,U,U2,E2)))*Pjt(t,i,L,N,U,U2,E2)
    return det(matprod)
end


################################
### HCB Correlation Function ###
################################
function C(t::Float64,L::Int64,N::Int64,U::Matrix{Float64},U2::Matrix{Float64},E2::Vector{Float64},parity::Bool,TI::Bool)
    """
    Calculate LxL equal-time one-body correlation matrix for HCB.

    INPUTS:
    L: number of lattice sites
    N: number of particles
    U: LxL matrix of single particle components

    OUTPUTS:
    LxL matrix of equal-time one-body correlations. 
    """
    Ct0::Matrix{ComplexF64} = NCorrZeroT(N,U)
    Cmat::Matrix{ComplexF64} = Diagonal(diag(C_SFt(t,U2,E2,Ct0)))
    if TI==true
        Threads.@threads for j in range(2,L)
            Cmat[1,j] = Gijt(1,j,t,L,N,U,U2,E2)
        end
        for j in range(2,L-1)
            Cmat[j,j+1:L] = Cmat[j-1,j:L-1]
        end
    elseif parity==true
        partitions = Iterators.Stateful(chunks(L,24))
        tasks = map(partitions) do chunk 
            Threads.@spawn for i::Int in chunk
                for j::Int in range(i+1,L-(i-1))
                    Cmat[i,j] = Gijt(i,j,t,L,N,U,U2,E2)
                end
                Cmat[i+1:L-i,L-(i-1)] = reverse(conj.(Cmat[i,i+1:L-i]))
            end
        end
        fetch.(tasks)
    else
        Threads.@threads for i::Int in range(1,L)
            for j::Int in range(1,L)
                if i != j
                    Cmat[i,j] = Gijt(i,j,t,L,N,U,U2,E2)
                end
            end
        end
    end
    return Hermitian(Cmat)
end

function main(L::Int64,Nb::Int64,V::Float64,t::Float64)
    #################
    ### load data ###
    #################
    sites::Array{Float64,1} = range(0,L-1,length=L);
    E::Vector{Float64} = eigvals(TrapHamiltonian(L,1.0,0.1,V,true))
    U::Matrix{Float64} = eigvecs(TrapHamiltonian(L,1.0,0.1,V,true))
    E2::Vector{Float64} = eigvals(FreeHamiltonian(L,1.0,0.1,true))
    U2::Matrix{Float64} = eigvecs(FreeHamiltonian(L,1.0,0.1,true))
    # E::Vector{Float64} = eigvals(BraggHamiltonian(L,1.0,0.1,0.0,20,pi/4,false))
    # U::Matrix{Float64} = eigvecs(BraggHamiltonian(L,1.0,0.1,0.0,20,pi/4,false))

    xi::Float64 = 1/sqrt(V)
    println(string("The characteristic denisty is ",Nb/xi))
    # println("Gijt:")
    # @time Gijt(1,2,t,L,Nb,U,U2,E2)
    # println("Saving eigenvector data")
    # open(string("spectrum/L=",L,"_V=",V,"_trap_PBC_energy.bin"),"w") do f
    #    write(f,E)
    # end

    println("OBDM:")
    @time C_HCB::Matrix{ComplexF64} =  C(t,L,Nb,U,U2,E2,true,false) 
    # open(string("C_T=0_NonEq/C_L=",L,"_N=",Nb,"_V=",V,"_t=",t,"_trap_PBC.bin"),"w") do f
    #     write(f,C_HCB)
    # end
    # println("MDF:")
    # @time n_HCBL::Vector{Float64} = real(BLAS.map(k->nkt(k,float(L),C_HCB,sites),range(-pi,pi,L+1)));
    # open(string("C_T=0_NonEq/n_L=",L,"_N=",Nb,"_V=",V,"_t=",t,"_trap_PBC_Lnorm.bin"),"w") do f
    #     write(f,n_HCBL)
    # end
    # @time n_HCBxi::Vector{Float64} = real(BLAS.map(k->nkt(k,xi,C_HCB,sites),range(-pi,pi,L+1)));
    # open(string("C_T=0_NonEq/n_L=",L,"_N=",Nb,"_V=",V,"_t=",t,"_trap_PBC_xinorm.bin"),"w") do f
    #     write(f,n_HCBxi)
    # end
    # println(string("t=",t," done."))
end

main(10,3,1e-2,1.0)

# for t in range(10,100,10)
#     main(400,31,5*1e-4,t)
# end