using LinearAlgebra
using NLsolve
using Optim
using NonlinearSolve 
using Roots
using Distributed


###########################################
### Zero Temperature Correlation Matrix ###
###########################################
function NCorrZeroT(N::Int, U::Matrix{Float64})
    """
    Calculate N-body correlation matrix in equilibrium at zero temperature.

    INPUTS:
    N: number of particles
    U: matrix of single-particle equilibrium eigenstates

    OUTPUTS:
    C: matrix of one-body correlations
    """
    P = U[:,1:N]
    C::Matrix{Float64} = P * transpose(P)
    return Symmetric(C)
end


#############################################
### Finite Temperature Correlation Matrix ###
#############################################
function NCorrFiniteT(L::Int, beta::Float64, U::Matrix{Float64}, E::Vector{Float64}, mu::Float64)
    """
    Calculate N-body correlation function in equilibrium at finite temperature

    INPUTS
    L: number of sites
    beta: inverse temperature
    U: matrix of eigenvectors
    E: vector of eigenvalues
    mu: chemical potential

    OUTPUT
    C: matrix of one-body correlations
    """
    identity::Matrix{Float64} = Matrix(1.0I,L,L)
    D::Matrix{Float64} = inv(identity + Diagonal(exp.(-beta * (E - mu*ones(L)))))
    Dprime::Matrix{Float64} = U * D * transpose(U)
    # Dprime2::Matrix{Float64} = U * Dprime
    C::Matrix{Float64} = identity - Dprime
    return Symmetric(C)
end

function GetChemicalPotential(L::Int64, beta::Float64, N::Int64, U::Matrix{Float64}, E::Vector{Float64})
    """
    Get chemical potential provided N particles at temperature T

    INPUTS
    L: number of lattice sites; int
    N: number of particles; int
    beta: inverse temperature; float
    U: matrix of eigenvectors; matrix of floats
    E: list of eigenvalues; list of floats

    OUTPUT
    mu: chemical potental; float
    """
    func(mu) = sum(diag(NCorrFiniteT(L,beta,U,E,mu))) - N
    mu_guess = maximum(E[1:N])
    sol = find_zero(func,mu_guess)
    return sol
end


#######################
### Time Dependence ###
#######################
function C_SFt(t::Float64,U2::Matrix{Float64},E2::Vector{Float64},C0::Matrix{Float64})
    """
    Calculate equal-time one-body density matrix at non-zero time after a quench.
    
    INPUTS
    t: time; units of hbar/J
    beta: inverse temperature; units of 1/J
    mu: chemical potential; units of J
    L: lattice sites; dimensionless
    U2: matrix of eigenvectors of post-quench Hamiltonian
    E2: vector of eigenvalues of post-quench Hamiltonian
    C0: Initial one-body correlation function, can be zero T or finite T.

    OUTPUT
    One-body correlation density matrix.
    """
    D = Diagonal(exp.(complex(0,t)*E2))
    D1 = U2 * D * transpose(U2)
    C2 = adjoint(D1) * C0 * D1
    return Symmetric(C2)
end


##############################################
### Calculate MDF Given Correlation Matrix ###
##############################################
function nkt(k::Float64, L::Float64, C::Matrix{Float64}, sites::Vector{Float64})
    """
    Calculate momentum distribution function at finite T and non-zero time after a quench.
    
    INPUTS
    k: quasi-momentum; units of 1/a
    L: lattice sites; dimensionless
    C: correlation matrix
    sites: vector of L lattice sites

    OUTPUT
    Momentum distribution function
    """
    A = exp.(sites*complex(0,k))
    eik = A * transpose(conj(A)) 
    return sum(eik .* C) / L
end