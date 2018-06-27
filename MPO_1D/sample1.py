import numpy as np
from Physical_operators import *
from MPO_1D import MPO_1D

def model_hamiltonian(h, J, V):
  H_exact = np.zeros((8, 8), dtype=np.complex64)
  # VSz*I*Sz + JSz*Sz + hSx

  H_exact[0][0] = 3*h; H_exact[0][1] = 0; H_exact[0][2] = 0; H_exact[0][3] = J; H_exact[0][4] = 0; H_exact[0][5] = V; H_exact[0][6] = J; H_exact[0][7] =   0;
  H_exact[1][0] =   0; H_exact[1][1] = h; H_exact[1][2] = J; H_exact[1][3] = 0; H_exact[1][4] = V; H_exact[1][5] = 0; H_exact[1][6] = 0; H_exact[1][7] =   J;
  H_exact[2][0] =   0; H_exact[2][1] = J; H_exact[2][2] = h; H_exact[2][3] = 0; H_exact[2][4] = J; H_exact[2][5] = 0; H_exact[2][6] = 0; H_exact[2][7] =   V;
  H_exact[3][0] =   J; H_exact[3][1] = 0; H_exact[3][2] = 0; H_exact[3][3] =-h; H_exact[3][4] = 0; H_exact[3][5] = J; H_exact[3][6] = V; H_exact[3][7] =   0;
  H_exact[4][0] =   0; H_exact[4][1] = V; H_exact[4][2] = J; H_exact[4][3] = 0; H_exact[4][4] = h; H_exact[4][5] = 0; H_exact[4][6] = 0; H_exact[4][7] =   J;
  H_exact[5][0] =   V; H_exact[5][1] = 0; H_exact[5][2] = 0; H_exact[5][3] = J; H_exact[5][4] = 0; H_exact[5][5] =-h; H_exact[5][6] = J; H_exact[5][7] =   0;
  H_exact[6][0] =   J; H_exact[6][1] = 0; H_exact[6][2] = 0; H_exact[6][3] = V; H_exact[6][4] = 0; H_exact[6][5] = J; H_exact[6][6] =-h; H_exact[6][7] =   0;
  H_exact[7][0] =   0; H_exact[7][1] = J; H_exact[7][2] = V; H_exact[7][3] = 0; H_exact[7][4] = J; H_exact[7][5] = 0; H_exact[7][6] = 0; H_exact[7][7] =-3*h;

  return H_exact

#exact Sz for check without dissipation
def exact_Sz_average(t, calculate_index, h, J, V):
  if(calculate_index == 1):
    H_exact = np.zeros((8, 8), dtype=np.complex64)
    # VSz*I*Sz + JSz*Sz + hSx
    H_exact = model_hamiltonian(h, J, V)

    S, V = np.linalg.eigh(H_exact)
    V = np.transpose(V, (1, 0))

    Sz_3 = np.diag([1.0, 1.0/3.0, 1.0/3.0, -1.0/3.0, 1.0/3.0, -1.0/3.0, -1.0/3.0, -1.0])

    v3 = V[0][7]*np.exp(1j*S[0]*t)*V[0]
    v4 = V[0][7]*np.exp(1j*S[0]*t)*np.dot(Sz_3, V[0])
    for i in range(1, 8):
      v3 = v3 + V[i][7]*np.exp(1j*S[i]*t)*V[i]
      v4 = v4 + V[i][7]*np.exp(1j*S[i]*t)*np.dot(Sz_3, V[i])

    return np.dot(np.conjugate(v3), v4).real
  else:
    return

#exact Sz for check without dissipation
def exact_SzSz_correlation_average(t, calculate_index, h, J, V, dist=1):
  if(calculate_index == 1):
    H_exact = np.zeros((8, 8), dtype=np.complex64)
    # VSz*I*Sz + JSz*Sz + hSx
    H_exact = model_hamiltonian(h, J, V)

    S, V = np.linalg.eigh(H_exact)
    V = np.transpose(V, (1, 0))

    if(dist == 1):
      SzSz_3 = np.diag([1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0])
    elif(dist == 2):
      SzSz_3 = np.diag([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0])

    v3 = V[0][7]*np.exp(1j*S[0]*t)*V[0]
    v4 = V[0][7]*np.exp(1j*S[0]*t)*np.dot(SzSz_3, V[0])
    for i in range(1, 8):
      v3 = v3 + V[i][7]*np.exp(1j*S[i]*t)*V[i]
      v4 = v4 + V[i][7]*np.exp(1j*S[i]*t)*np.dot(SzSz_3, V[i])

    return np.dot(np.conjugate(v3), v4).real
  else:
    return

#exact Sz for check without dissipation
def exact_SxSx_correlation_average(t, calculate_index, h, J, V, dist=1):
  if(calculate_index == 1):
    H_exact = np.zeros((8, 8), dtype=np.complex64)
    # VSz*I*Sz + JSz*Sz + hSx
    H_exact = model_hamiltonian(h, J, V)

    S, V = np.linalg.eigh(H_exact)
    V = np.transpose(V, (1, 0))

    SxSx_3 = np.zeros((8, 8), dtype=np.complex64)
    if(dist == 1):
      SxSx_3[0][3] = 1.0; SxSx_3[0][6] = 1.0;
      SxSx_3[1][2] = 1.0; SxSx_3[1][7] = 1.0;
      SxSx_3[2][1] = 1.0; SxSx_3[2][4] = 1.0;
      SxSx_3[3][0] = 1.0; SxSx_3[3][5] = 1.0;
      SxSx_3[4][7] = 1.0; SxSx_3[4][2] = 1.0;
      SxSx_3[5][6] = 1.0; SxSx_3[5][3] = 1.0;
      SxSx_3[6][5] = 1.0; SxSx_3[6][0] = 1.0;
      SxSx_3[7][4] = 1.0; SxSx_3[7][1] = 1.0;
      SxSx_3 = SxSx_3 / 2.0
    elif(dist == 2):
      SxSx_3[0][5] = 1.0;
      SxSx_3[1][4] = 1.0;
      SxSx_3[2][7] = 1.0;
      SxSx_3[3][6] = 1.0;
      SxSx_3[4][1] = 1.0;
      SxSx_3[5][0] = 1.0;
      SxSx_3[6][3] = 1.0;
      SxSx_3[7][2] = 1.0;

    v3 = V[0][7]*np.exp(1j*S[0]*t)*V[0]
    v4 = V[0][7]*np.exp(1j*S[0]*t)*np.dot(SxSx_3, V[0])
    for i in range(1, 8):
      v3 = v3 + V[i][7]*np.exp(1j*S[i]*t)*V[i]
      v4 = v4 + V[i][7]*np.exp(1j*S[i]*t)*np.dot(SxSx_3, V[i])

    return np.dot(np.conjugate(v3), v4).real
  else:
    return

#time evolution of density matrix with dissipation
class density_matrix_time_evolution:
  def __init__(self, T_delta, h, J, coef):
    self.density_matrix = np.zeros((8, 8), dtype=np.complex64)
    self.density_matrix[7][7] = 1.0
    self.T_delta = T_delta
    self.h = h
    self.J = J
    self.coef = coef
    #H = hSz + JSx*Sx + VSx*I*Sx
    self.Hamiltonian = np.zeros((8, 8), dtype=np.complex64)
    self.Hamiltonian = model_hamiltonian(h, J, V)
    # K = Sx + iSy
    self.K1 = np.zeros((8, 8), dtype=np.complex64)
    self.K1[0][1] = 1.0; self.K1[2][3] = 1.0; self.K1[4][5] = 1.0; self.K1[6][7] = 1.0;
    self.K1_dagger = np.transpose(self.K1)
    self.K1K1 = np.dot(self.K1_dagger, self.K1)

    self.K2 = np.zeros((8, 8), dtype=np.complex64)
    self.K2[0][2] = 1.0; self.K2[1][3] = 1.0; self.K2[4][6] = 1.0; self.K2[5][7] = 1.0;
    self.K2_dagger = np.transpose(self.K2)
    self.K2K2 = np.dot(self.K2_dagger, self.K2)

    self.K3 = np.zeros((8, 8), dtype=np.complex64)
    self.K3[0][4] = 1.0; self.K3[1][5] = 1.0; self.K3[2][6] = 1.0; self.K3[3][7] = 1.0;
    self.K3_dagger = np.transpose(self.K3)
    self.K3K3 = np.dot(self.K3_dagger, self.K3)

    self.Sz = np.zeros((8, 8), dtype=np.complex64)
    self.Sz[0][0] = 3.0; self.Sz[1][1] = 1.0; self.Sz[2][2] = 1.0; self.Sz[3][3] = -1.0; self.Sz[4][4] = 1.0; self.Sz[5][5] = -1.0; self.Sz[6][6] = -1.0; self.Sz[7][7] = -3.0;
    self.Sz = self.Sz/3.0
  
  def normalize(self):
    trace = np.trace(self.density_matrix)
    self.density_matrix = self.density_matrix/trace

  def update(self):
    diff = -1j*(np.dot(self.Hamiltonian, self.density_matrix) - np.dot(self.density_matrix, self.Hamiltonian))
    diff = diff + self.coef*self.coef*(np.dot(self.K1, np.dot(self.density_matrix, self.K1_dagger)) - 0.5*np.dot(self.K1K1, self.density_matrix) - 0.5*np.dot(self.density_matrix, self.K1K1))
    diff = diff + self.coef*self.coef*(np.dot(self.K2, np.dot(self.density_matrix, self.K2_dagger)) - 0.5*np.dot(self.K2K2, self.density_matrix) - 0.5*np.dot(self.density_matrix, self.K2K2))
    diff = diff + self.coef*self.coef*(np.dot(self.K3, np.dot(self.density_matrix, self.K3_dagger)) - 0.5*np.dot(self.K3K3, self.density_matrix) - 0.5*np.dot(self.density_matrix, self.K3K3))
    self.density_matrix = self.density_matrix + diff*T_delta
    self.normalize()

  def calculate_Sz(self):
    return np.trace(np.dot(self.Sz, self.density_matrix))

  def calculate_entanglement_entropy(self):
    reduced_density_matrix = np.zeros((2, 2), dtype=np.complex64)
    reduced_density_matrix[0][0] = self.density_matrix[0][0] + self.density_matrix[1][1] + self.density_matrix[2][2] + self.density_matrix[3][3]
    reduced_density_matrix[0][1] = self.density_matrix[0][4] + self.density_matrix[1][5] + self.density_matrix[2][6] + self.density_matrix[3][7]
    reduced_density_matrix[1][0] = self.density_matrix[4][0] + self.density_matrix[5][1] + self.density_matrix[6][2] + self.density_matrix[7][3]
    reduced_density_matrix[1][1] = self.density_matrix[4][4] + self.density_matrix[5][5] + self.density_matrix[6][6] + self.density_matrix[7][7]

    S = np.linalg.eigvalsh(reduced_density_matrix)
    return -(S[0]*np.log(S[0].real) + S[1]*np.log(S[1].real))

if __name__ == '__main__':
  N=3
  D=8
  J=1.0
  h=1.0
  V=1.0
  coef = 1.0
  T_init=0
  T_end=10
  T_delta=1.0E-3

  TEST = MPO_1D(N=N, D=D, T_delta=T_delta)
  TEST2 = density_matrix_time_evolution(T_delta=T_delta, h=h, J=J, coef=coef)

  print("initialize density matrix")
  TEST.initialize_density_matrix()

  print("initial <Sz>")
  print(TEST.calculate_expectation_of_1body(Sz))

  filename = "sample1.out"
  f = open(filename, "w")
  f.close()

  TEST.initialize_H_1body()
  TEST.add_H_1body(Sz, h)
  TEST.initialize_K_1body()
  TEST.add_K_1body(Sx+1j*Sy, coef/2.0)

  TEST.initialize_H_2body()
  TEST.add_H_2body(Sx, Sx, J)
  TEST.initialize_H_3body()
  TEST.add_H_3body(Sx, I, Sx, V)

  t = T_init
  while(t < T_end):
    #operate 1-body interactions
    for n in range(N):
      TEST.operate_1body(n)

    #operate 2-body interactions
    for n in range(int(N/2)):
      TEST.operate_2body(2*n, 2*n+1)
    for n in range(int((N-1)/2)):
      TEST.operate_2body(2*n+1, 2*n+2)

    #operate 3-body interactions
    for n in range(int(N/3)):
      TEST.operate_3body(3*n, 3*n+1, 3*n+2)
    for n in range(int((N-1)/3)):
      TEST.operate_3body(3*n+1, 3*n+2, 3*n+3)
    for n in range(int((N-2)/3)):
      TEST.operate_3body(3*n+2, 3*n+3, 3*n+4)

    TEST2.update()

    TEST.normalize()

    Sz_ave_array = TEST.calculate_expectation_of_1body(O=Sz)
    Sz_cor1_array = TEST.calculate_expectation_of_2body(O1=Sx, O2=Sx, dist=1)
    Sz_cor2_array = TEST.calculate_expectation_of_2body(O1=Sx, O2=Sx, dist=1)
    Sz_ave = sum(Sz_ave_array).real/len(Sz_ave_array)
    SzSz1_ave = sum(Sz_cor1_array).real/len(Sz_cor1_array)
    SzSz2_ave = sum(Sz_cor2_array).real/len(Sz_cor2_array)

    Sz_exact = exact_Sz_average(t-T_init, 1, h, J, V)
    SxSx1_exact = exact_SxSx_correlation_average(t-T_init, 1, h, J, V, dist=1)
    SxSx2_exact = exact_SxSx_correlation_average(t-T_init, 1, h, J, V, dist=1)

    Sz_exact2 = TEST2.calculate_Sz()

    Entanglement_Entropy = TEST.calculate_entanglement_entropy(2, 2).real
    Entanglement_Entropy_exact = TEST2.calculate_entanglement_entropy().real

    f = open(filename, "a")
    print("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n" % (t, Sz_ave, Sz_exact, Sz_exact2, SzSz1_ave, SxSx1_exact, SzSz2_ave, SxSx2_exact, Entanglement_Entropy, Entanglement_Entropy_exact))
    f.write("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n" % (t, Sz_ave, Sz_exact, Sz_exact2, SzSz1_ave, SxSx1_exact, SzSz2_ave, SxSx2_exact, Entanglement_Entropy, Entanglement_Entropy_exact))
    f.close()

    if(np.abs(Sz_ave) > 2.0):
      break

    t = t + T_delta

