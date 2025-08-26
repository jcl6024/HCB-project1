"""
16/07/25 V1
24/07/25 V2
19/08/25 V3

Zero Temperature Equilibrium Momentum Distribution Function for Hard Core Bosons.
"""


#########################
### Import Statements ###
#########################
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
    Id = Vector{Float64}(undef,L)
    Id = ones(L)
    BLAS.scal!(j-1,-1.0,Id,stride(Id,1))
    return Id
end

# function Oij(i::Int,j::Int,L::Int)::Vector{Float64}
#     """
#     Return a diagonal matrix wherein the first j-1 entries are -1 and all
#     others 1.

#     INPUTS
#     j: site index
#     L: number of lattice sites
 
#     OUTPUT
#     LxL diagonal matrix.
#     """
#     Id::Vector{Float64} = ones(L)
#     BLAS.scal!(abs(i-j),-1.0,Id,stride(Id,1))
#     return Id
# end


function Delta(i::Int,j::Int,L::Int)::Matrix{Float64}
    """
    Write matrix representation of kronecker delat ij.

    INPUTS
    j: site index
    L: number of sites
    """
    ZeroMat = Matrix{Float64}(undef,L,L)

    ZeroMat = zeros(Float64,L,L)
    ZeroMat[i,j] = 1.0
    return ZeroMat
end

function boltzmann_factor(L::Int,beta::Float64,mu::Float64,E::Vector{Float64})::Matrix{Float64}
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
    one = Vector{Float64}(undef,L)
    BF = Matrix{Float64}(undef,L,L)

    one = ones(Int,L)
    # diag(BF) = exp.(-beta*(E - mu*one))
    return Diagonal(exp.(-beta*(E - mu*one)))
end


function Gij(i::Int64,j::Int64,L::Int64,U::Matrix{Float64},BF::Matrix{Float64},invZ::Matrix{Float64})::Float64
    """
    Construct i,j component of one-body density matrix.

    INPUTS
    i,j: site indices
    L: number of lattice sites
    N: number of particles
    U: unitary matrix of components of single particle eigenstates
    BF: diagonal matrix of boltzmann factors (BF) for each single-particle eigenvalue
    invZ: inverse matrix of (Id + BF); the determinant of this matrix yields the partition function Z
    A,B,D: LxL placeholderzero matrices 

    OUTPUTS
    Lx(N+1) matrix of components of single-particle eigenstates after action
    of Jordan-Wigner strings and particle creation at site j.
    """
    local A = Matrix{Float64}(undef,L,L)
    local B = Matrix{Float64}(undef,L,L)
    local D = Matrix{Float64}(undef,L,L)
    # local C = Matrix{Float64}(undef,L,L)

    local det1::Float64
    local det2::Float64

    D = Delta(i,j,L)

    BLAS.gemm!('T','N',1.0,U,O(i,L).*O(j,L).*U,0.0,A)
    BLAS.gemm!('N','N',1.0,invZ,A+BF,0.0,B)
    det1 = det(B)

    BLAS.gemm!('N','N',1.0,D,U,0.0,B)
    BLAS.gemm!('T','N',1.0,U,B,0.0,D)
    BLAS.gemm!('N','N',1.0,invZ,D+A+BF,0.0,B)
    det2 = det(B)

    return (-1)^(i-j) * (det2 - det1)
end


################################
### HCB Correlation Function ###
################################
function C(L::Int64,N::Int64,beta::Float64,mu::Float64,U::Matrix{Float64},E::Vector{Float64},BF::Matrix{Float64},parity::Bool,TI::Bool)::Matrix{Float64}
    """
    Calculate LxL equal-time one-body correlation matrix for HCB.

    INPUTS:
    L: number of lattice sites
    N: number of particles
    beta: inverse temperature
    U: LxL matrix of single particle components
    E: L dim vector of single particle eigenenergies
    BF: diagonal matrix of Boltzmann factors
    parity: is system parity symmetric? true or false
    TI: is system translationally invariant? true or false (NOTE: if system is also parity symmetric
        do NOT put true for both, only put true for TI)

    OUTPUTS:
    LxL matrix of equal-time one-body correlations. 
    """
    Cmat = Matrix{Float64}(undef,L,L)
    Id = Matrix{Float64}(undef,L,L)
    invz = Matrix{Float64}(undef,L,L)

    Cmat = NCorrFiniteT(L, beta, U, E, mu)
    Id = Matrix(1.0I,L,L)
    invz = inv(Id + BF)
    if TI==true
        Threads.@threads for j in range(2,L)
            Cmat[1,j] = Gij(j,1,L,U,BF,invz)
        end
        for j in range(2,L-1)
            Cmat[j,j+1:L] = Cmat[j-1,j:L-1]
        end
    elseif parity==true
        Threads.@threads for i::Int in range(1,L/2)
            for j::Int in range(i+1,L-(i-1))
                Cmat[i,j] = Gij(j,i,L,U,BF,invz)
            end
            Cmat[i+1:L-i,L-(i-1)] = reverse(Cmat[i,i+1:L-i])
        end
    else
        Threads.@threads for i::Int in range(1,L)
            for j::Int in range(1,L)
                if i < j
                    Cmat[i,j] = Gij(j,i,L,U,BF,invz)
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
    V::Float64 = 0
    E::Vector{Float64} = eigvals(FreeHamiltonian(L,1.0,0.1,false))
    U::Matrix{Float64} = eigvecs(FreeHamiltonian(L,1.0,0.1,false))
    # E::Vector{Float64} = eigvals(TrapHamiltonian(L,1.0,0.0,1e-4,true))
    # U::Matrix{Float64} = eigvecs(TrapHamiltonian(L,1.0,0.0,1e-4,true))
    # E::Vector{Float64} = eigvals(BraggHamiltonian(L,1.0,0.1,0.0,20,pi/4,false))
    # U::Matrix{Float64} = eigvecs(BraggHamiltonian(L,1.0,0.1,0.0,20,pi/4,false))

    ###############
    ### Outputs ###
    ###############
    xi::Float64 = L # 1/sqrt(V)
    # println(string("The characteristic denisty is ",Nb/xi))
    mu::Float64 = GetChemicalPotential(L, T, Nb, U, E)
    BF::Matrix{Float64} = boltzmann_factor(L,T,mu,E) 
    Id::Matrix{Float64} = Matrix(1.0I,L,L)
    invz::Matrix{Float64} = inv(Id + BF)
    # @time Gij(1,2,L,U,BF,invz)
    # print(0)

    ############################
    ### compute OBDM and MDF ###
    ############################
    println("HCB OBDM:")
    @time C_HCB::Matrix{Float64} = C(L,Nb,T,mu,U,E,BF,true,false)
    open("C_FiniteT_Eq/C/C_L=$(L)_N=$(Nb)_beta=$(T)_V=$(V)_free_PBC.bin","w") do f
        write(f,C_HCB)
    end
    println("HCB MDF:")
    @time n_HCBxi::Vector{Float64} = real(BLAS.map(k->nkt(k,xi,C_HCB,sites),range(-pi,pi,L+1)));
    open("C_FiniteT_Eq/n/n_L=$(L)_N=$(Nb)_beta=$(T)_V=$(V)_free_PBC.bin","w") do f
        write(f,n_HCBxi)
    end
    print(0)
end

# main(200,100,100.0)

for T::Float64 in [2.0,10.0,100.0]
    main(500,200,T)
end