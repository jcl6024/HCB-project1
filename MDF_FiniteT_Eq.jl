"""
16/07/25 V1
24/07/25 V2

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
using LinearAlgebra
using BenchmarkTools
###############
### Include ###
###############
include("Hamiltonian_functions.jl")
include("SF_pure_functions.jl")
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
    Id::Vector{Float64} =  ones(L)
    BLAS.scal!(j-1,-1.0,Id,stride(Id,1))
    return Diagonal(Id)
end

function Delta(i::Int,j::Int,L::Int)
    """
    Write matrix representation of kronecker delat ij.

    INPUTS
    j: site index
    L: number of sites
    """
    ZeroMat::Matrix{Float64} = zeros(Float64,L,L)
    ZeroMat[i,j] = 1.0
    return ZeroMat
end

function boltzmann_factor(L::Int,beta::Float64,mu::Float64,U::Matrix{Float64},E::Vector{Float64})
    """
    Construct grand canonical Boltzmann factor exp(-beta(H-mu N)) for a system
    of hard-core bosons in thermal equilibrium at temperature T.

    INPUTS
    L: number of sites
    beta: inverse temperature
    mu: chemical potential
    U: matrix of single-particle wavefunctions
    E: vector of single-particle eigenvalues
    """
    # one = ones(Int,L)
    # D::Matrix{Float64} = Diagonal(exp.(-beta*(E - mu*one)))
    # return BF::Matrix{Float64} = U * D * adjoint(U)
    one = ones(Int,L)
    D::Matrix{Float64} = exp.(-beta*(E - mu*one)) .* transpose(U)
    return BLAS.gemm('N','N',U,D)
end

# function partition_function(L::Int,BF::Matrix{Float64})
#     """
#     Calculate grand canonical partition function.

#     INPUTS
#     L: number of sites
#     BF: boltzmann factor
#     """
#     return det(Matrix(1I,L,L)+BF)
# end

function Pij(i::Int64,j::Int64,L::Int64,BF::Matrix{Float64})
    """
    Construct i,j component of one-body density matrix.

    INPUTS
    j: site index
    L: number of lattice sites
    N: number of particles
    U: unitary matrix of components of single particle eigenstates

    OUTPUTS
    Lx(N+1) matrix of components of single-particle eigenstates after action
    of Jordan-Wigner strings and particle creation at site j.
    """
    Id::Matrix{Float64} = Matrix(1.0I,L,L)
    BF_ij::Matrix{Float64} = O(j,L)*BF*O(i,L)
    invZ::Matrix{Float64} = inv(Id + BF)
    A1::Matrix{Float64} = Id + BLAS.gemm('N','N',(Id + Delta(i,j,L)), BF_ij)
    A2::Matrix{Float64} = Id + BF_ij
    A1Z::Matrix{Float64} = BLAS.gemm('N','N',A1, invZ)
    A2Z::Matrix{Float64} = BLAS.gemm('N','N',A2, invZ)
    return (det(A1Z) - det(A2Z))
end

################################
### HCB Correlation Function ###
################################
function C(L::Int64,N::Int64,beta::Float64,mu::Float64,U::Matrix{Float64},E::Vector{Float64},BF::Matrix{Float64},parity::Bool,TI::Bool)
    """
    Calculate LxL equal-time one-body correlation matrix for HCB.

    INPUTS:
    L: number of lattice sites
    N: number of particles
    U: LxL matrix of single particle components

    OUTPUTS:
    LxL matrix of equal-time one-body correlations. 
    """
    Cmat::Matrix{Float64} = Diagonal(diag(NCorrFiniteT(L, beta, U, E, mu)))
    if TI==true
        Threads.@threads for j in range(2,L)
            Cmat[1,j] = Pij(1,j,L,BF)
        end
        for j in range(2,L-1)
            Cmat[j,j+1:L] = Cmat[j-1,j:L-1]
        end
    elseif parity==true
        partitions = Iterators.Stateful(chunks(L,24))
        tasks = map(partitions) do chunk 
            Threads.@spawn for i::Int in chunk
                for j::Int in range(i+1,L-(i-1))
                    Cmat[i,j] = Pij(i,j,L,BF)
                end
                Cmat[i+1:L-i,L-(i-1)] = reverse(Cmat[i,i+1:L-i])
            end
        end
        fetch.(tasks)
    else
        Threads.@threads for i::Int in range(1,L)
            for j::Int in range(1,L)
                if i != j
                    Cmat[i,j] = Pij(i,j,L,BF)
                end
            end
        end
    end
    return Symmetric(Cmat)
end

function main(L::Int64,Nb::Int64,T::Float64)
    #################
    ### load data ###
    #################
    sites::Array{Float64,1} = range(0,L-1,length=L);

    E::Vector{Float64} = eigvals(FreeHamiltonian(L,1.0,0.1,true))
    U::Matrix{Float64} = eigvecs(FreeHamiltonian(L,1.0,0.1,true))
    
    # V::Float64 = 3.3*1e-4
    # println("eigenvalue time:")
    # E::Vector{Float64} = eigvals(TrapHamiltonian(L,1.0,0.0,1e-4,true))
    # println("eigenvector time:")
    # U::Matrix{Float64} = eigvecs(TrapHamiltonian(L,1.0,0.0,1e-4,true))
    
    # E::Vector{Float64} = eigvals(BraggHamiltonian(L,1.0,0.1,0.0,20,pi/4,false))
    # U::Matrix{Float64} = eigvecs(BraggHamiltonian(L,1.0,0.1,0.0,20,pi/4,false))

    ###############
    ### Outputs ###
    ###############
    xi::Float64 = L
    println(string("The characteristic denisty is ",Nb/xi))
    mu::Float64 = GetChemicalPotential(L, T, Nb, U, E)
    @time BF::Matrix{Float64} = boltzmann_factor(L,T,mu,U,E)
    # @time Pij(328,507,L,BF)
    # @time Delta(55,690,1000)

    println("HCB OBDM:")
    @time C_HCB::Matrix{Float64} = C(L,Nb,T,mu,U,E,BF,false,true) 
    open("C_FiniteT_Eq/C_L=$(L)_N=$(Nb)_beta=$(T)_free_PBC.bin","w") do f
        write(f,C_HCB)
    end
    println("HCB MDF:")
    @time n_HCBxi::Vector{Float64} = real(BLAS.map(k->nkt(k,xi,C_HCB,sites),range(-pi,pi,L+1)));
    open("C_FiniteT_Eq/n_L=$(L)_N=$(Nb)_beta=$(T)_free_PBC.bin","w") do f
        write(f,n_HCBxi)
    end
end

for t::Float64 in [2.0,10.0,100.0]
    main(1000,501,t)
end