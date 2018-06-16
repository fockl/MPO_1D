import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
import copy
from Physical_operators import *

class MPO_1D:
  def __init__(self, N=2, D=10, T_init=-10, T_end=10, T_delta=1.0E-3, calculate_index=1):
    self.C = C
    self.T_init = T_init
    self.T_end = T_end
    self.T_delta = T_delta
    self.N = N

    self.calculate_index = calculate_index
    self.D = D

    #self.left_edge_vector = np.ones(D, dtype=np.complex64)/np.sqrt(D)
    #self.right_edge_vector = np.ones(D, dtype=np.complex64)/np.sqrt(D)

    self.H_self = np.zeros((2, 2), dtype=np.complex64)
    self.K_self = np.zeros((2, 2), dtype=np.complex64)
    self.H_2body = np.zeros((4, 4), dtype=np.complex64)
    self.H_3body = np.zeros((8, 8), dtype=np.complex64)
    self.L_self = np.zeros((((2, 2, 2, 2))), dtype=np.complex64)
    self.L_2body = np.zeros((((4, 4, 4, 4))), dtype=np.complex64)
    self.L_3body = np.zeros((((8, 8, 8, 8))), dtype=np.complex64)

    self.p_set = np.zeros(((((self.N, D, 2, 2, D)))), dtype=np.complex64)

  def __Liouvillian_self(self):
    self.L_self = np.zeros((((2, 2, 2, 2))), dtype=np.complex64)
    H_transpose = np.transpose(self.H_self)
    K_transpose = np.transpose(self.K_self)
    K_conjugate = np.conjugate(self.K_self)
    K_dagger = np.conjugate(K_transpose)
    K1 = np.dot(K_dagger, self.K_self)
    K2 = np.dot(K_conjugate, K_transpose)

    for i in range(2):
      for j in range(2):
        for k in range(2):
          for l in range(2):
            self.L_self[i][j][k][l] = -1j*(self.H_self[i][k]*I2[j][l] - I2[i][k]*H_transpose[j][l])
            self.L_self[i][j][k][l] = self.L_self[i][j][k][l] + self.K_self[i][k]*K_conjugate[j][l]
            self.L_self[i][j][k][l] = self.L_self[i][j][k][l] - 0.5*K1[i][k]*I2[j][l] - 0.5*I2[i][k]*K2[j][l]

  def __Liouvillian_2body(self):
    self.L_2body = np.zeros((((4, 4, 4, 4))), dtype=np.complex64)
    H_transpose = np.transpose(self.H_2body)

    for i in range(4):
      for j in range(4):
        for k in range(4):
          for l in range(4):
            self.L_2body[i][j][k][l] = -1j*(self.H_2body[i][k]*I4[j][l] - I4[i][k]*H_transpose[j][l])

  def __Liouvillian_3body(self):
    self.L_3body = np.zeros((((8, 8, 8, 8))), dtype=np.complex64)
    H_transpose = np.transpose(self.H_3body)

    for i in range(8):
      for j in range(8):
        for k in range(8):
          for l in range(8):
            self.L_3body[i][j][k][l] = -1j*(self.H_3body[i][k]*I8[j][l] - I8[i][k]*H_transpose[j][l])

  def set_H_self(self, H_self):
    self.H_self = H_self
    self.__Liouvillian_self()

  def set_K_self(self, K_self):
    self.K_self = K_self
    self.__Liouvillian_self()

  def set_H_2body(self, H_2body):
    self.H_2body = H_2body
    self.__Liouvillian_2body()

  def set_H_3body(self, H_3body):
    self.H_3body = H_3body
    self.__Liouvillian_3body()

  def trace_of_density_matrix(self):
    v_set = np.zeros(((self.N, D, D)), dtype=np.complex64)
    for p_num in range(self.N):
      for D1 in range(D):
        for D2 in range(D):
          for i in range(2):
            v_set[p_num][D1][D2] = v_set[p_num][D1][D2] + self.p_set[p_num][D1][i][i][D2]

    #v_left  = copy.deepcopy(self.left_edge_vector)
    product_v = np.zeros((D, D), dtype=np.complex64)
    #for p_num in range(self.N):
    #  v_left = np.dot(v_left, v_set[p_num])
    for p_num in range(self.N-1):
      product_v = np.dot(v_set[p_num], v_set[p_num+1])
      v_set[p_num+1] = product_v

    #return np.dot(v_left, self.left_edge_vector).real
    return np.trace(product_v).real

  def normalize(self):
    p_trace = self.trace_of_density_matrix()
    p_trace = np.power(p_trace, 1.0/self.N)
    TEST.p_set = TEST.p_set/p_trace

  def initialize_density_matrix(self):
    self.p_set = np.zeros(((((self.N, D, 2, 2, D)))), dtype=np.complex64)
    for n in range(N):
      for D1 in range(D):
        for D2 in range(D):
          self.p_set[n][D1][1][1][D2] = 1.0

    self.normalize()

  def calculate_Sz_average(self):
    p_set_0 = np.zeros(((self.N, D, D)), dtype=np.complex64)
    p_set_1 = np.zeros(((self.N, D, D)), dtype=np.complex64)
    for n in range(self.N):
      for D1 in range(D):
        for D2 in range(D):
          for d in range(2):
            p_set_0[n][D1][D2] = p_set_0[n][D1][D2] + self.p_set[n][D1][d][d][D2]

      p_tmp = np.transpose(self.p_set[n], (1, 0, 2, 3))
      p_tmp = np.reshape(p_tmp, (2, (D*2*D)))
      p_tmp = np.dot(Sz, p_tmp)
      p_tmp = np.reshape(p_tmp, (2, D, 2, D))
      p_tmp = np.transpose(p_tmp, (1, 0, 2, 3))

      for D1 in range(D):
        for D2 in range(D):
          for d in range(2):
            p_set_1[n][D1][D2] = p_set_1[n][D1][D2] + p_tmp[D1][d][d][D2]

    calc_Sz = np.zeros(self.N, dtype=np.complex64)

    #p_product_left = np.zeros((self.N, D), dtype=np.complex64)
    #p_product_right = np.zeros((self.N, D), dtype=np.complex64)

    p_product_left = np.zeros(((self.N, D, D)), dtype=np.complex64)
    p_product_right = np.zeros(((self.N, D, D)), dtype=np.complex64)

    #v_left  = copy.deepcopy(self.left_edge_vector)
    #v_right = copy.deepcopy(self.right_edge_vector)

    #p_product_left[0] = np.dot(v_left, p_set_0[0])
    #p_product_right[self.N-1] = np.dot(p_set_0[self.N-1], v_right)
    p_product_left[0] = p_set_0[0]
    p_product_right[self.N-1] = p_set_0[self.N-1]

    for n in range(1, self.N):
      p_product_left[n] = np.dot(p_product_left[n-1], p_set_0[n])
      p_product_right[self.N-1-n] = np.dot(p_set_0[self.N-1-n], p_product_right[self.N-n])

    #calc_Sz[0] = np.dot(np.dot(v_left, p_set_1[0]), p_product_right[1])
    calc_Sz[0] = np.trace(np.dot(p_set_1[0], p_product_right[1]))
    for n in range(1, self.N-1):
      #calc_Sz[n] = np.dot(p_product_left[n-1], np.dot(p_set_1[n], p_product_right[n+1]))
      calc_Sz[n] = np.trace(np.dot(p_product_left[n-1], np.dot(p_set_1[n], p_product_right[n+1])))
    #calc_Sz[self.N-1] = np.dot(p_product_left[self.N-2], np.dot(p_set_1[self.N-1], v_right))
    calc_Sz[self.N-1] = np.trace(np.dot(p_product_left[self.N-2], p_set_1[self.N-1]))

    sum_Sz = 0.0
    for n in range(self.N):
      sum_Sz = sum_Sz + calc_Sz[n].real

    return sum_Sz/self.N

  def __decompose_L_2body_to_p(self, L):
    p1, S0, p2 = svd(L)
    ## svds have bad precision

    p1 = p1[:,:D]
    S0 = S0[:D]
    p2 = p2[:D,:]

    S0 = np.diag(np.sqrt(S0))
    p1 = np.dot(p1, S0)
    p2 = np.dot(S0, p2)
    p1 = np.reshape(p1, (D, 2, 2, D))
    p2 = np.reshape(p2, (D, 2, 2, D))

    return p1, p2

  def __decompose_L_3body_to_p(self, L):
    p1, S0, L2 = svd(L)
    ## svds have bad precision

    p1 = p1[:,:D]
    S0 = S0[:D]
    L2 = L2[:D,:]

    S0 = np.diag(np.sqrt(S0))
    p1 = np.dot(p1, S0)
    L2 = np.dot(S0, L2)
    p1 = np.reshape(p1, (D, 2, 2, D))
    L2 = np.reshape(L2, (D*2*2, 2*2*D))
    p2, p3 = self.__decompose_L_2body_to_p(L2)

    return p1, p2, p3

  def operate_self(self, index):
    #operate 1-body interactions
    print("operate self")
    L_self = copy.deepcopy(self.L_self)
    L_self = np.reshape(L_self, (4, 4))
    L_self = I4 + L_self*self.T_delta

    p_tmp = np.transpose(self.p_set[index], (1, 2, 0, 3))
    p_tmp = np.reshape(p_tmp, (2*2, D*D))
    p_tmp = np.reshape(np.dot(L_self, p_tmp), (2, 2, D, D))
    self.p_set[index] = np.transpose(p_tmp, (2, 0, 1, 3))

  def operate_2body(self, index1, index2):
    #operate 2-body interactions
    print("operate 2body")
    L_2body = copy.deepcopy(self.L_2body)
    L_2body = np.reshape(L_2body, (4*4, 4*4))

    p_tmp1 = np.reshape(self.p_set[index1], (D*2*2, D))
    p_tmp2 = np.reshape(self.p_set[index2], (D, 2*2*D))

    P = np.dot(p_tmp1, p_tmp2)
    P = np.reshape(P, (D, 2, 2, 2, 2, D))
    P = np.transpose(P, (1, 3, 2, 4, 0, 5))
    P = np.reshape(P, (2*2*2*2, D*D))

    L_2body = I16 + L_2body*self.T_delta
    L_2body = np.dot(L_2body, P)

    L_2body = np.reshape(L_2body, (2, 2, 2, 2, D, D))
    L_2body = np.transpose(L_2body, (4, 0, 2, 1, 3, 5))
    L_2body = np.reshape(L_2body, (D*2*2, 2*2*D))

    p_tmp1, p_tmp2 = self.__decompose_L_2body_to_p(L_2body)

    self.p_set[index1] = np.reshape(p_tmp1, (D, 2, 2, D))
    self.p_set[index2] = np.reshape(p_tmp2, (D, 2, 2, D))

  def operate_3body(self, index1, index2, index3):
    #operate 3-body interactions
    print("operate 3body")
    L_3body = copy.deepcopy(self.L_3body)
    L_3body = np.reshape(L_3body, (4*4*4, 4*4*4))

    p_tmp1 = np.reshape(self.p_set[index1], (D*2*2, D))
    p_tmp2 = np.reshape(self.p_set[index2], (D, 2*2*D))
    P = np.dot(p_tmp1, p_tmp2)
    P = np.reshape(P, (D*2*2*2*2, D))
    p_tmp3 = np.reshape(self.p_set[index3], (D, 2*2*D))
    P = np.dot(P, p_tmp3)
    P = np.reshape(P, (D, 2, 2, 2, 2, 2, 2, D))
    P = np.transpose(P, (1, 3, 5, 2, 4, 6, 0, 7))
    P = np.reshape(P, (2*2*2*2*2*2, D*D))

    L_3body = I64 + L_3body*self.T_delta
    L_3body = np.dot(L_3body, P)

    L_3body = np.reshape(L_3body, (2, 2, 2, 2, 2, 2, D, D))
    L_3body = np.transpose(L_3body, (6, 0, 3, 1, 4, 2, 5, 7))
    L_3body = np.reshape(L_3body, (D*2*2, 2*2*2*2*D))

    p_tmp1, p_tmp2, p_tmp3 = self.__decompose_L_3body_to_p(L_3body)

    self.p_set[index1] = np.reshape(p_tmp1, (D, 2, 2, D))
    self.p_set[index2] = np.reshape(p_tmp2, (D, 2, 2, D))
    self.p_set[index3] = np.reshape(p_tmp3, (D, 2, 2, D))

###############################################################################

def Hamiltonian_self(t, h, calculate_index=1):
  if(calculate_index == 1):
    return h*Sx
  elif(calculate_index == 2):
    return -t*Sz + Sx

def Hamiltonian_2body(t, J):
  H = np.zeros((2, 2, 2, 2), dtype=np.complex64)
  for i0 in range(2):
    for i1 in range(2):
      for i2 in range(2):
        for i3 in range(2):
          H[i0][i1][i2][i3] = Sz[i0][i2]*Sz[i1][i3]
  H = np.reshape(H, (4, 4))
  return -J*H

def Hamiltonian_3body(t, V):
  H = np.zeros((2, 2, 2, 2, 2, 2), dtype=np.complex64)
  for i0 in range(2):
    for i1 in range(2):
      for i2 in range(2):
        for i3 in range(2):
          for i4 in range(2):
            for i5 in range(2):
              H[i0][i1][i2][i3][i4][i5] = Sz[i0][i3]*I2[i1][i4]*Sz[i2][i5]
  H = np.reshape(H, (8, 8))
  return -V*H

def K():
  return np.zeros((4, 4), dtype=np.complex64)

def exact_Sz_average(t, calculate_index, h, J, V):
  if(calculate_index == 1):
    H_exact = np.zeros((8, 8), dtype=np.complex64)
    # -VSz*I*Sz -JSz*Sz + hSx
    H_exact[0][0] =-2.0*J-V; H_exact[0][1] =  h; H_exact[0][2] =      h; H_exact[0][3] =  0; H_exact[0][4] =  h; H_exact[0][5] =      0; H_exact[0][6] =  0; H_exact[0][7] =       0;
    H_exact[1][0] =       h; H_exact[1][1] =  V; H_exact[1][2] =      0; H_exact[1][3] =  h; H_exact[1][4] =  0; H_exact[1][5] =      h; H_exact[1][6] =  0; H_exact[1][7] =       0;
    H_exact[2][0] =       h; H_exact[2][1] =  0; H_exact[2][2] =2.0*J-V; H_exact[2][3] =  h; H_exact[2][4] =  0; H_exact[2][5] =      0; H_exact[2][6] =  h; H_exact[2][7] =       0;
    H_exact[3][0] =       0; H_exact[3][1] =  h; H_exact[3][2] =      h; H_exact[3][3] =  V; H_exact[3][4] =  0; H_exact[3][5] =      0; H_exact[3][6] =  0; H_exact[3][7] =       h;

    H_exact[4][0] =       h; H_exact[4][1] =  0; H_exact[4][2] =      0; H_exact[4][3] =  0; H_exact[4][4] =  V; H_exact[4][5] =      h; H_exact[4][6] =  h; H_exact[4][7] =       0;
    H_exact[5][0] =       0; H_exact[5][1] =  h; H_exact[5][2] =      0; H_exact[5][3] =  0; H_exact[5][4] =  h; H_exact[5][5] =2.0*J-V; H_exact[5][6] =  0; H_exact[5][7] =       h;
    H_exact[6][0] =       0; H_exact[6][1] =  0; H_exact[6][2] =      h; H_exact[6][3] =  0; H_exact[6][4] =  h; H_exact[6][5] =      0; H_exact[6][6] =  V; H_exact[6][7] =       h;
    H_exact[7][0] =       0; H_exact[7][1] =  0; H_exact[7][2] =      0; H_exact[7][3] =  h; H_exact[7][4] =  0; H_exact[7][5] =      h; H_exact[7][6] =  h; H_exact[7][7] =-2.0*J-V;

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

if __name__ == '__main__':
  N=3
  D=4
  J=0.5
  h=1.0
  V=-1.0
  T_init=-10
  T_end=0 + 1.0E-3
  T_delta=1.0E-3
  C=0.0
  calculate_index=1

  TEST = MPO_1D(N=N, D=D, T_init=T_init, T_end=T_end, T_delta=T_delta, calculate_index=calculate_index)

  print("initialize density matrix")
  TEST.initialize_density_matrix()

  print("initial <Sz>")
  print(TEST.calculate_Sz_average())

  filename = "Result.txt"
  f = open(filename, "w")
  f.close()

  TEST.set_H_self(Hamiltonian_self(0, h))
  TEST.set_K_self(K())
  TEST.set_H_2body(Hamiltonian_2body(0, J))
  TEST.set_H_3body(Hamiltonian_3body(0, V))
  
  t = T_init
  while(t < T_end):
    #operate 1-body interactions
    for n in range(N):
      TEST.operate_self(n)

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

    TEST.normalize()

    Sz_ave = TEST.calculate_Sz_average()
    Sz_exact = exact_Sz_average(t-T_init, calculate_index, h, J, V)

    f = open(filename, "a")
    print("%lf %lf %lf\n" % (t, Sz_ave, Sz_exact))
    f.write("%lf %lf %lf\n" % (t, Sz_ave, Sz_exact))
    f.close()

    if(np.abs(Sz_ave) > 2.0):
      break

    t = t + T_delta

