"""
28/07/25 V1

Zero Temperature Dynamic Momentum Distribution Function for Hard Core Bosons Following a Bragg Scattering Pulse Quench.
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

function Dt(E::Vector{Float64},t::Float64)
    """
    Construct diagonal matrix representation of time evolution operator
    in the basis of eigenvalues E.

    INPUTS
    E: vector of eigenvalues
    t: time

    OUTPUT
    diagonal matrix of dimension dim(E).
    """
    return Diagonal(exp.(complex(0,-t)*E))
end

function Pt(tau::Float64,t::Float64,L::Int64,N::Int64,U::Matrix{Float64},E::Vector{Float64},U2::Matrix{Float64},E2::Vector{Float64})
    """
    Construct the matrix P of coefficients of an initial state (assuming
    ground state) with additional row corresponding to creation of a particle
    at the site j and corresponding signs owing to Jordan-Wigner strings. Time 
    evolve according to Bragg pulse quench.

    INPUTS
    tau: duration of Bragg scattering pulse
    t: time variable
    j: site index
    L: number of lattice sites
    N: number of particles
    U: unitary matrix of components of single particle eigenstates pre-quench
    U2: unitary matrix of components of single particle Bragg eigenstates
    E2: vector of Bragg eigenenergies

    OUTPUTS
    Lx(N+1) matrix of components of single-particle eigenstates after action
    of Jordan-Wigner strings and particle creation at site j.
    """
    Ptau::Matrix{ComplexF64} = U2 * Dt(E2,tau) * adjoint(U2) * U[:,1:N]
    Pt::Matrix{ComplexF64} = U * Dt(E,t) * adjoint(U) * Ptau
    return Pt
end

function PjtB(j::Int64,L::Int,Pt::Matrix{ComplexF64})
    """
    Construct the matrix P of coefficients of an initial state (assuming
    ground state) with additional row corresponding to creation of a particle
    at the site j and corresponding signs owing to Jordan-Wigner strings. Time 
    evolve according to Bragg pulse quench.

    INPUTS
    tau: duration of Bragg scattering pulse
    t: time variable
    j: site index
    L: number of lattice sites
    N: number of particles
    U: unitary matrix of components of single particle eigenstates pre-quench
    U2: unitary matrix of components of single particle Bragg eigenstates
    E2: vector of Bragg eigenenergies

    OUTPUTS
    Lx(N+1) matrix of components of single-particle eigenstates after action
    of Jordan-Wigner strings and particle creation at site j.
    """
    colj::Vector{Int} = zeros(Int,L)
    colj[j] = 1
    return  O(j,L)*([Pt;;colj])::Matrix{ComplexF64}
end

function Gijt(i::Int64,j::Int64,L::Int,Pt::Matrix{ComplexF64})
    """
    Calculate one-body Green's function for i!=j.

    INPUTS
    i,j: site index
    tau: duration of Bragg scattering pulse
    t: time
    L: number of lattice sites
    N: number of particles
    U: unitary matrix of components of single particle eigenstates pre-quench
    U2: unitary matrix of components of single particle Bragg eigenstates
    E2: vector of Bragg eigenenergies

    OUTPUT
    ji entry of the correlation matrix.
    """
    matprod = adjoint(PjtB(j,L,Pt))*PjtB(i,L,Pt)
    return det(matprod)
end


################################
### HCB Correlation Function ###
################################
function C(tau::Float64,t::Float64,L::Int64,N::Int64,U::Matrix{Float64},E::Vector{Float64},U2::Matrix{Float64},E2::Vector{Float64},parity::Bool,TI::Bool)
    """
    Calculate LxL equal-time one-body correlation matrix for HCB.

    INPUTS:
    tau: duration of Bragg scattering pulse
    t: time
    L: number of lattice sites
    N: number of particles
    U: LxL matrix of single particle components
    U2: unitary matrix of components of single particle Bragg eigenstates
    E2: vector of Bragg eigenenergies
    parity: system is inversion symmetric true or false (almost always false)
    TI: system is translationally invriant true of false (almost always false)

    OUTPUTS:
    LxL matrix of equal-time one-body correlations. 
    """
    Ct0::Matrix{ComplexF64} = NCorrZeroT(N,U)
    Ctau::Matrix{ComplexF64} = C_SFt(tau,U2,E2,Ct0)
    Cmat::Matrix{ComplexF64} = Diagonal(diag(C_SFt(t,U,E,Ctau)))
    P::Matrix{ComplexF64} = Pt(tau,t,L,N,U,E,U2,E2)
    if TI==true
        Threads.@threads for j in range(2,L)
            Cmat[1,j] = Gijt(1,j,L,P)
        end
        for j in range(2,L-1)
            Cmat[j,j+1:L] = Cmat[j-1,j:L-1]
        end
    elseif parity==true
        partitions = Iterators.Stateful(chunks(L,24))
        tasks = map(partitions) do chunk 
            Threads.@spawn for i::Int in chunk
                for j::Int in range(i+1,L-(i-1))
                    Cmat[i,j] = Gijt(i,j,L,P)
                end
                Cmat[i+1:L-i,L-(i-1)] = reverse(conj.(Cmat[i,i+1:L-i]))
            end
        end
        fetch.(tasks)
    else
        Threads.@threads for i::Int in range(1,L)
            for j::Int in range(1,L)
                if i < j
                    Cmat[i,j] = Gijt(i,j,L,P)
                end
            end
        end
    end
    return Hermitian(Cmat)
end

function main(L::Int64,Nb::Int64,V::Float64,tau::Float64,t::Float64)
    #################
    ### load data ###
    #################
    sites::Array{Float64,1} = range(0,L-1,length=L);
      
    xi::Float64 = 1/sqrt(V)
    E::Vector{Float64} = eigvals(TrapHamiltonian(L,1.0,0.1,V,true))
    U::Matrix{Float64} = eigvecs(TrapHamiltonian(L,1.0,0.1,V,true))

    # xi::Float = L
    # E::Vector{Float64} = eigvals(FreeHamiltonian(L,1.0,0.1,true))
    # U::Matrix{Float64} = eigvecs(FreeHamiltonian(L,1.0,0.1,true))

    E2::Vector{Float64} = eigvals(BraggHamiltonian(L,1.0,0.1,V,20,pi/4,true))
    U2::Matrix{Float64} = eigvecs(BraggHamiltonian(L,1.0,0.1,V,20,pi/4,true))

    println(string("The characteristic denisty is ",Nb/xi))
    println("Gijt:")
    Gijt(Int(L/2),Int(L/2)+1,tau,t,L,Nb,U,E,U2,E2)

    println("OBDM:")
    @time C_HCB::Matrix{ComplexF64} =  C(tau,t,L,Nb,U,E,U2,E2,false,false) 
    open("T=0_Bragg/C/C_L=$(L)_N=$(Nb)_V=$(V)_tau=$(tau)_t=$(t)_trap_bragg_PBC.bin","w") do f
        write(f,C_HCB)
    end
    println("MDF:")
    @time n_HCB::Vector{Float64} = real(BLAS.map(k->nkt(k,xi,C_HCB,sites),range(-pi,pi,L+1)));
    open("T=0_Bragg/n/n_L=$(L)_N=$(Nb)_V=$(V)_tau=$(tau)_t=$(t)_trap_bragg_PBC.bin","w") do f
        write(f,n_HCB)
    end
    println(string("t=",t," done."))
end

for t::Float64 in range(0,5,11)
    main(500,31,1e-4,0.1,t)
end