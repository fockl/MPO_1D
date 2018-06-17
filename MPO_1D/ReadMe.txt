MPO_1D is a class to calculate time evolution of density matrix with MPO

MPO_1D(N, D, T_delta, H_1body_kind=1, K_1body_kind=1, H_2body_kind=1, H_3body_kind=1)
  explanation:
    prepare MPO_1D class

  variable:
    N : The number of 1/2 spins
    D : Bond dimention of MPO
    T_delta : time discritization

  optional:
    H_1body_kind : The kinds of 1-body Hamiltonian
    K_1body_kind : The kinds of 1-body Dissipation
    H_2body_kind : The kinds of 2-body Hamiltonian
    H_3body_kind : The kinds of 3-body Hamiltonian

void initialize_density_matrix():
  explanation:
    initialize density matrix

  variable:

  optional:

void initialize_H_1body(index=-1)
  explanation:
    initialize 1-body Hamiltonian

  variable:

  optional:
    index : The kind of 1-body Hamiltonian initialized
            if negative, all 1-body Hamiltonians are initialized

void initialize_K_1body(index=-1)
  explanation:
    initialize 1-body Dissipation

  variable:

  optional:
    index : The kind of 1-body Dissipation initialized
            if negative, all 1-body Dissipations are initialized

void initialize_H_2body(index=-1)
  explanation:
    initialize 2-body Hamiltonian

  variable:

  optional:
    index : The kind of 2-body Hamiltonian initialized
            if negative, all 2-body Hamiltonians are initialized


void initialize_H_3body(index=-1)
  explanation:
    initialize 3-body Hamiltonian

  variable:

  optional:
    index : The kind of 3-body Hamiltonian initialized
            if negative, all 3-body Hamiltonians are initialized

void add_H_1body(H_1body, coef=1.0, index=0):
  explanation:
    add 1-body Hamiltonian

  variable:
    H_1body : Hamiltonian Operator with shape (2, 2) you add

  optional:
    coef : coefficient of 1-body Hamiltonian
    index : The kind of 1-body Hamiltonian you add to

void add_K_1body(K_1body, coef=1.0, index=0):
  explanation:
    add 1-body Dissipation

  variable:
    K_1body : Dissipation Operator with shape (2, 2) you add

  optional:
    coef : coefficient of 1-body Dissipation
    index : The kind of 1-body Dissipation you add to

void add_H_2body(H_2body1, H_2body2, coef=1.0, index=0):
  explanation:
    add 2-body nearest neighbor Hamiltonian

  variable:
    H_2body1, H_2body2 : Hamiltonian Operators with shape (2, 2) you add
    added Hamiltonian is H_2body1 \dot H_2body2 interaction

  optional:
    coef : coefficient of 2-body Hamiltonian
    index : The kind of 2-body Hamiltonian you add to

void add_H_3body(H_3body1, H_3body2, H_3body3, coef=1.0, index=0):
  explanation:
    add 3-body nearest neighbor Hamiltonian

  variable:
    H_3body1, H_3body2, H_3body3 : Hamiltonian Operators with shape (2, 2) you add
    added Hamiltonian is H_3body1 \dot H_3body2 \dot H_3body3 interaction

  optional:
    coef : coefficient of 3-body Hamiltonian
    index : The kind of 3-body Hamiltonian you add to

void normalize():
  explanation:
    normalize density matrix so that Trace of is to be 1

  variable:

  optional:

np.float64 trace_of_density_matrix():
  explanation:
    calculate trace of density matrix

  variable:

  optional:

np.array(dtype=np.complex64)/np.complex64 calculate_expectation_of_1body(self, O=Sz, index=-1)
  explanation:
    calculate expectation value of given 1-body operator at each site

  variable:

  optional:
    O : 1-body operator with shape (2, 2) you want to calculate expectation. Default is Sz.
    index : calculate expectation value at that site and return complex
            if negative, expectation values are calculated at all sites and return array

np.array(dtype=np.complex64)/np.complex64 calculate_expectation_of_2body(self, O1=Sz, O2=Sz, dist=1, index=-1)
  explanation:
    calculate expectation value of given 2-body operator at each sites with distance dist
    Default settings gives Sz_i \dot Sz_{i+1} 

  variable:

  optional:
    O1, O2 : 1-body operators with shape (2, 2) you want to calculate expectation. Default is Sz.
    dist : the distance 2 operators are operated
    index : calculate expectation value at that site and return complex
            if negative, expectation values are calculated at all sites and return array

void operate_1body(particle_index, Liouvillian_index=0)
  explanation:
    operate 1body liouvillian at particle_index site

  variable:
    particle_index : the site where liouvillian operate

  optional:
    liouvillian_index : the kind of liouvillian operated

void operate_2body(particle_index1, particle_index2, Liouvillian_index=0)
  explanation:
    operate 2body liouvillian at particle_index sites

  variable:
    particle_index1, particle_index2 : the sites where liouvillian operate.
    particle_index1+1 = particle_index2

  optional:
    liouvillian_index : the kind of liouvillian operated

void operate_3body(particle_index1, particle_index2, particle_index3, Liouvillian_index=0)
  explanation:
    operate 3body liouvillian at particle_index sites

  variable:
    particle_index1, particle_index2, particle_index3 : the sites where liouvillian operate.
    particle_index1+2 = particle_index2+1 = particle_index3

  optional:
    liouvillian_index : the kind of liouvillian operated

