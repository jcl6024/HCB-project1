"""
16/07/25 V1

Zero Temperature Equilibrium Momentum Distribution Function for Hard Core Bosons.
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
using BenchmarkTools
###############
### Include ###
###############
include("Hamiltonian_functions.jl")
include("SF_functions.jl")

function main(L::Int64,Nb::Int64)

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

    function Pj(j::Int64,L::Int64,N::Int64,U::Matrix{Float64})
        """
        Construct the matrix P of coefficients of an initial state (assuming
        ground state) with additional row corresponding to creation of a particle
        at the site j and corresponding signs owing to Jordan-Wigner strings.

        INPUTS
        j: site index
        L: number of lattice sites
        N: number of particles
        U: unitary matrix of components of single particle eigenstates

        OUTPUTS
        Lx(N+1) matrix of components of single-particle eigenstates after action
        of Jordan-Wigner strings and particle creation at site j.
        """
        colj::Vector{Int} = zeros(Int,L)
        colj[j] = 1
        P::Matrix{Float64} = O(j,L)*[U[:,1:N];;colj]
        return P
    end

    function Gij(i::Int64,j::Int64,L::Int64,N::Int64,U::Matrix{Float64})
        """
        Calculate one-body Green's function for i!=j.

        INPUTS
        j: site index
        L: number of lattice sites
        N: number of particles
        U: unitary matrix of components of single particle eigenstates

        OUTPUT
        ji entry of the correlation matrix.
        """
        matprod::Matrix{Float64} = adjoint(Pj(i,L,N,U))*Pj(j,L,N,U)
        return det(matprod)
    end

    #################
    ### load data ###
    #################
    # L::Int64 = 300
    sites::Array{Float64,1} = range(0,L-1,length=L);
    E::Vector{Float64} = eigvals(FreeHamiltonian(L,1.0,0.1,true))
    U::Matrix{Float64} = eigvecs(FreeHamiltonian(L,1.0,0.1,true))
    # E::Vector{Float64} = eigvals(TrapHamiltonian(L,1.0,0.1,2e-5,false))
    # U::Matrix{Float64} = eigvecs(TrapHamiltonian(L,1.0,0.1,2e-5,false))
    # E::Vector{Float64} = eigvals(BraggHamiltonian(L,1.0,0.1,0.0,20,pi/4,false))
    # U::Matrix{Float64} = eigvecs(BraggHamiltonian(L,1.0,0.1,0.0,20,pi/4,false))


    ################################
    ### HCB Correlation Function ###
    ################################
    function C(L::Int64,N::Int64,U::Matrix{Float64},parity::Bool)
        """
        Calculate LxL equal-time one-body correlation matrix for HCB.

        INPUTS:
        L: number of lattice sites
        N: number of particles
        U: LxL matrix of single particle components

        OUTPUTS:
        LxL matrix of equal-time one-body correlations. 
        """
        Cmat::Matrix{Float64} = Diagonal(diag(NCorrZeroT(N,U)))
        if parity==true
            Threads.@threads for i in range(1,L)
                Cmat[i,i+1:L-(i-1)] = BLAS.map(j->Gij(i,j,L,N,U),range(i+1,L-(i-1)))
            end
            Threads.@threads for i in range(1,L)
                Cmat[i+1:L-i,L-(i-1)] = reverse(Cmat[i,i+1:L-i])
            end
        else
            Threads.@threads for i in range(1,L)
                Cmat[i,i+1:L] =  BLAS.map(j->Gij(i,j,L,N,U),range(i+1,L))
            end
        end
        return Symmetric(Cmat)
    end


    ###############
    ### Outputs ###
    ###############
    # Nb::Int = 31
    # @time Gij(1,50,L,31,U);
    @time C_HCB::Matrix{Float64} = C(L,Nb,U,true)
    open(string("C_T=0_Equilibrium/C_L=200_N=21_free_PBC.bin"),"w") do f
        write(f,C_HCB)
    end
    @time n_HCB::Vector{Float64} = real(BLAS.map(k->nkt(k,L,C_HCB,sites),range(-pi,pi,L+1)));
    open(string("C_T=0_Equilibrium/n_L=",L,"_N=",Nb,"_free_PBC.bin"),"w") do f
        write(f,n_HCB)
    end
    print(0)
end

main(300,31)