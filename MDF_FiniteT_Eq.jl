"""
16/07/25 V1
24/07/25 V2

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
function O(j::Int,L::Int)::Vector{Float64}
    """
    Return a diagonal matrix wherein the first j-1 entries are -1 and all
    others 1.

    INPUTS
    j: site index
    L: number of lattice sites
 
    OUTPUT
    LxL diagonal matrix.
    """
    Id::Vector{Float64} = ones(L)
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
    ZeroMat::Matrix{Float64} = zeros(Float64,L,L)
    # ZeroMat = Matrix{Float64}(undef,L,L)
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
    # one = ones(Int,L)
    # D::Matrix{Float64} = Diagonal(exp.(-beta*(E - mu*one)))
    # return BF::Matrix{Float64} = U * D * adjoint(U)
    one = ones(Int,L)
    return  Diagonal(exp.(-beta*(E - mu*one)))
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

function Pij(i::Int64,j::Int64,L::Int64,BF::Matrix{Float64},Id::Matrix{Float64},invZ::Matrix{Float64})
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
    BF_ij::Matrix{Float64} = O(j,L).*BF.*O(i,L)
    A1::Matrix{Float64} = Id + BLAS.gemm('N','N',(Id + Delta(i,j,L)), BF_ij)
    A2::Matrix{Float64} = Id + BF_ij
    A1Z::Matrix{Float64} = BLAS.gemm('N','N',A1, invZ)
    A2Z::Matrix{Float64} = BLAS.gemm('N','N',A2, invZ)
    # pij::Float64 = det(A1) - det(A2)
    return (det(A1Z) - det(A2Z))::Float64 
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
    local C = Matrix{Float64}(undef,L,L)

    BLAS.gemm!('T','N',1.0,U,O(i,L).*O(j,L).*U,0.0,A)
    det1::Float64 = det(BLAS.gemm('N','N',invZ,A+BF))

    BLAS.gemm!('N','N',1.0,Delta(i,j,L),U,0.0,B)
    BLAS.gemm!('T','N',1.0,U,B,0.0,C)
    det2::Float64 = det(BLAS.gemm('N','N',invZ,A+C+BF))

    # BLAS.gemm!('N','N',1.0,D,U,0.0,B)
    # BLAS.gemm!('T','N',1.0,U,B,0.0,D)

    # @time transpose(U)*Delta(i,j,L)*U
    # @time dij::Matrix{Float64} = BLAS.gemm('N','N',Delta(i,j,L),U)
    # @time dij = BLAS.gemm('T','N',U,dij)

    # @time AijZ::Matrix{Float64} = BLAS.gemm('N','N',invZ,Aij)
    # @time dijZ::Matrix{Float64} = BLAS.gemm('N','N',invZ,Aij + dij)
    # @time det_Aij::Float64 = det(AijZ)
    # @time det_dij::Float64 = det(dijZ)

    # dij::Matrix{Float64} = O(j,L)*O(i,L)+Delta(i,j,L)
    # dij = BLAS.gemm('N','N',dij,U)
    # dij = BLAS.gemm('T','N',U,dij)
    # dij = BLAS.gemm('N','N',invZ,(dij + BF))
    # det_dij::Float64 = det(dij)

    # dij2::Matrix{Float64} = BLAS.gemm('N','N',dij,U)
    # dij3::Matrix{Float64} = BLAS.gemm('T','N',U,dij2)
    # dij4::Matrix{Float64} = BLAS.gemm('N','N',(dij3 + BF),invZ)

    # Aij2::Matrix{Float64} = BLAS.gemm('T','N',U,Aij)
    # Aij3::Matrix{Float64} = BLAS.gemm('N','N',Aij2,U)
    # Aij4::Matrix{Float64} = BLAS.gemm('N','N',Aij3 + BF,invZ)

    # ((-1).^(i-j)) * (det(BLAS.gemm('N','N',invZ,A+D+BF)) - det(BLAS.gemm('N','N',invZ,A+BF)))

    # (-1)^(i-j) * (det2 - det1)

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
    U: LxL matrix of single particle components

    OUTPUTS:
    LxL matrix of equal-time one-body correlations. 
    """
    Cmat::Matrix{Float64} = Diagonal(diag(NCorrFiniteT(L, beta, U, E, mu)))
    Id::Matrix{Float64} = Matrix(1.0I,L,L)
    invz::Matrix{Float64} = inv(Id + BF)
    A::Matrix{Float64} = zeros(Float64,L,L)
    # B::Matrix{Float64} = zeros(Float64,L,L)
    # D::Matrix{Float64} = zeros(Float64,L,L)
    if TI==true
        Threads.@threads for j in range(2,L)
            Cmat[1,j] = Gij(1,j,L,U,BF,Id,invz)
        end
        for j in range(2,L-1)
            Cmat[j,j+1:L] = Cmat[j-1,j:L-1]
        end
    elseif parity==true
        Threads.@threads for i::Int in range(1,L/2)
            for j::Int in range(i+1,L-(i-1))
                Cmat[i,j] = Gij(j,i,L,U,BF,invz)
                # Cmat[i,j] = i + j
            end
            Cmat[i+1:L-i,L-(i-1)] = reverse(Cmat[i,i+1:L-i])
        end
    else
        Threads.@threads for i::Int in range(1,L)
            for j::Int in range(1,L)
                if i < j
                    Cmat[i,j] = Pij(j,i,L,U,BF,Id,invz)
                end
            end
        end
    end
    return Hermitian(Cmat)
end

function main(L::Int64,Nb::Int64,T::Float64)
    #################
    ### load data ###
    #################
    sites::Array{Float64,1} = range(0,L-1,length=L);

    E::Vector{Float64} = eigvals(FreeHamiltonian(L,1.0,0.1,false))
    U::Matrix{Float64} = eigvecs(FreeHamiltonian(L,1.0,0.1,false))
    
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
    BF::Matrix{Float64} = boltzmann_factor(L,T,mu,E)
    Id::Matrix{Float64} = Matrix(1.0I,L,L)
    invz::Matrix{Float64} = inv(Id + BF)

    println("HCB OBDM:")
    @time C_HCB::Matrix{Float64} = C(L,Nb,T,mu,U,E,BF,true,false)
    # print(0)
    open("C_FiniteT_Eq/C/C_L=$(L)_N=$(Nb)_beta=$(T)_free_test.bin","w") do f
        write(f,C_HCB)
    end
    println("HCB MDF:")
    @time n_HCBxi::Vector{Float64} = real(BLAS.map(k->nkt(k,xi,C_HCB,sites),range(-pi,pi,L+1)));
    open("C_FiniteT_Eq/n/n_L=$(L)_N=$(Nb)_beta=$(T)_free_test.bin","w") do f
        write(f,n_HCBxi)
    end
end

main(200,101,100.0)

# for L::Int in range(100,600,6)
#     main(L,Int(L/2+1),10.0)
# end