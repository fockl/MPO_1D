import numpy as np
from scipy.linalg import svd
#from scipy.sparse.linalg import svds
import copy
from Physical_operators import *

class MPO_1D:
  def __init__(self, N=2, D=10, T_delta=1.0E-3):
    self.T_delta = T_delta
    self.N = N

    self.D = D

    self.left_edge_vector = np.ones(self.D, dtype=np.complex64)/np.sqrt(self.D)
    self.right_edge_vector = np.ones(self.D, dtype=np.complex64)/np.sqrt(self.D)

    self.H_self = np.zeros((2, 2), dtype=np.complex64)
    self.K_self = np.zeros((2, 2), dtype=np.complex64)
    self.H_2body = np.zeros((4, 4), dtype=np.complex64)
    self.H_3body = np.zeros((8, 8), dtype=np.complex64)
    self.L_self = np.zeros((((2, 2, 2, 2))), dtype=np.complex64)
    self.L_2body = np.zeros((((4, 4, 4, 4))), dtype=np.complex64)
    self.L_3body = np.zeros((((8, 8, 8, 8))), dtype=np.complex64)

    self.p_set = np.zeros(((((self.N, self.D, 2, 2, self.D)))), dtype=np.complex64)

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

  def initialize_H_self(self):
    self.H_self = np.zeros((2, 2), dtype=np.complex64)
    self.__Liouvillian_self()

  def add_H_self(self, H_self, coef=1.0):
    if(H_self.shape != (2, 2)):
      raise NameError("H_self type should be (2, 2)")
    H_tmp = np.zeros((2, 2), dtype=np.complex64)
    for i0 in range(2):
      for i1 in range(2):
        H_tmp[i0][i1] = H_self[i0][i1]
    self.H_self = self.H_self + coef*H_tmp
    self.__Liouvillian_self()

  def initialize_K_self(self):
    self.L_self = np.zeros((2, 2), dtype=np.complex64)
    self.__Liouvillian_self()

  def add_K_self(self, K_self, coef=1.0):
    if(K_self.shape != (2, 2)):
      raise NameError("K_self type should be (2, 2)")
    K_tmp = np.zeros((2, 2), dtype=np.complex64)
    for i0 in range(2):
      for i1 in range(2):
        K_tmp[i0][i1] = K_self[i0][i1]
    self.K_self = self.K_self + coef*K_tmp
    self.__Liouvillian_self()

  def initialize_H_2body(self):
    self.H_2body = np.zeros((4, 4), dtype=np.complex64)
    self.__Liouvillian_2body()

  def add_H_2body(self, H_2body1, H_2body2, coef=1.0):
    if(H_2body1.shape != (2, 2)):
      raise NameError("H_2body1 type should be (2, 2)")
    if(H_2body2.shape != (2, 2)):
      raise NameError("H_2body2 type should be (2, 2)")
    H_tmp = np.zeros((((2, 2, 2, 2))), dtype=np.complex64)
    for i0 in range(2):
      for i1 in range(2):
        for i2 in range(2):
          for i3 in range(2):
            H_tmp[i0][i1][i2][i3] = H_2body1[i0][i2]*H_2body2[i1][i3]
    H_tmp = np.reshape(H_tmp, (4, 4))
    self.H_2body = self.H_2body + coef*H_tmp
    self.__Liouvillian_2body()

  def initialize_H_3body(self):
    self.H_3body = np.zeros((8, 8), dtype=np.complex64)
    self.__Liouvillian_3body()

  def add_H_3body(self, H_3body1, H_3body2, H_3body3, coef=1.0):
    if(H_3body1.shape != (2, 2)):
      raise NameError("H_3body1 type should be (2, 2)")
    if(H_3body2.shape != (2, 2)):
      raise NameError("H_3body2 type should be (2, 2)")
    if(H_3body3.shape != (2, 2)):
      raise NameError("H_3body3 type should be (2, 2)")
    H_tmp = np.zeros((((((2, 2, 2, 2, 2, 2))))), dtype=np.complex64)
    for i0 in range(2):
      for i1 in range(2):
        for i2 in range(2):
          for i3 in range(2):
            for i4 in range(2):
              for i5 in range(2):
                H_tmp[i0][i1][i2][i3][i4][i5] = H_3body1[i0][i3]*H_3body2[i1][i4]*H_3body3[i2][i5]
    H_tmp = np.reshape(H_tmp, (8, 8))
    self.H_3body = self.H_3body + coef*H_tmp
    self.__Liouvillian_3body()

  def trace_of_density_matrix(self):
    v_set = np.zeros(((self.N, self.D, self.D)), dtype=np.complex64)
    for p_num in range(self.N):
      for D1 in range(self.D):
        for D2 in range(self.D):
          for i in range(2):
            v_set[p_num][D1][D2] = v_set[p_num][D1][D2] + self.p_set[p_num][D1][i][i][D2]

    v_left  = copy.deepcopy(self.left_edge_vector)
    product_v = np.zeros((self.D, self.D), dtype=np.complex64)
    for p_num in range(self.N):
      v_left = np.dot(v_left, v_set[p_num])
    #for p_num in range(self.N-1):
    #  product_v = np.dot(v_set[p_num], v_set[p_num+1])
    #  v_set[p_num+1] = product_v

    return np.dot(v_left, self.left_edge_vector).real
    #return np.trace(product_v).real

  def normalize(self):
    p_trace = self.trace_of_density_matrix()
    p_trace = np.power(p_trace, 1.0/self.N)
    self.p_set = self.p_set/p_trace

  def initialize_density_matrix(self):
    self.p_set = np.zeros(((((self.N, self.D, 2, 2, self.D)))), dtype=np.complex64)
    for n in range(self.N):
      for D1 in range(self.D):
        for D2 in range(self.D):
          self.p_set[n][D1][1][1][D2] = 1.0

    self.normalize()

  def calculate_Sz_average(self):
    p_set_0 = np.zeros(((self.N, self.D, self.D)), dtype=np.complex64)
    p_set_1 = np.zeros(((self.N, self.D, self.D)), dtype=np.complex64)
    for n in range(self.N):
      for D1 in range(self.D):
        for D2 in range(self.D):
          for d in range(2):
            p_set_0[n][D1][D2] = p_set_0[n][D1][D2] + self.p_set[n][D1][d][d][D2]

      p_tmp = np.transpose(self.p_set[n], (1, 0, 2, 3))
      p_tmp = np.reshape(p_tmp, (2, (self.D*2*self.D)))
      p_tmp = np.dot(Sz, p_tmp)
      p_tmp = np.reshape(p_tmp, (2, self.D, 2, self.D))
      p_tmp = np.transpose(p_tmp, (1, 0, 2, 3))

      for D1 in range(self.D):
        for D2 in range(self.D):
          for d in range(2):
            p_set_1[n][D1][D2] = p_set_1[n][D1][D2] + p_tmp[D1][d][d][D2]

    calc_Sz = np.zeros(self.N, dtype=np.complex64)

    p_product_left = np.zeros((self.N, self.D), dtype=np.complex64)
    p_product_right = np.zeros((self.N, self.D), dtype=np.complex64)

    #p_product_left = np.zeros(((self.N, self.D, self.D)), dtype=np.complex64)
    #p_product_right = np.zeros(((self.N, self.D, self.D)), dtype=np.complex64)

    v_left  = copy.deepcopy(self.left_edge_vector)
    v_right = copy.deepcopy(self.right_edge_vector)

    p_product_left[0] = np.dot(v_left, p_set_0[0])
    p_product_right[self.N-1] = np.dot(p_set_0[self.N-1], v_right)
    #p_product_left[0] = p_set_0[0]
    #p_product_right[self.N-1] = p_set_0[self.N-1]

    for n in range(1, self.N):
      p_product_left[n] = np.dot(p_product_left[n-1], p_set_0[n])
      p_product_right[self.N-1-n] = np.dot(p_set_0[self.N-1-n], p_product_right[self.N-n])

    calc_Sz[0] = np.dot(np.dot(v_left, p_set_1[0]), p_product_right[1])
    #calc_Sz[0] = np.trace(np.dot(p_set_1[0], p_product_right[1]))
    for n in range(1, self.N-1):
      calc_Sz[n] = np.dot(p_product_left[n-1], np.dot(p_set_1[n], p_product_right[n+1]))
      #calc_Sz[n] = np.trace(np.dot(p_product_left[n-1], np.dot(p_set_1[n], p_product_right[n+1])))
    calc_Sz[self.N-1] = np.dot(p_product_left[self.N-2], np.dot(p_set_1[self.N-1], v_right))
    #calc_Sz[self.N-1] = np.trace(np.dot(p_product_left[self.N-2], p_set_1[self.N-1]))

    sum_Sz = 0.0
    for n in range(self.N):
      sum_Sz = sum_Sz + calc_Sz[n].real

    return sum_Sz/self.N

  def __decompose_L_2body_to_p(self, L):
    p1, S0, p2 = svd(L)
    ## svds have bad precision

    p1 = p1[:,:self.D]
    S0 = S0[:self.D]
    p2 = p2[:self.D,:]

    S0 = np.diag(np.sqrt(S0))
    p1 = np.dot(p1, S0)
    p2 = np.dot(S0, p2)
    p1 = np.reshape(p1, (self.D, 2, 2, self.D))
    p2 = np.reshape(p2, (self.D, 2, 2, self.D))

    return p1, p2

  def __decompose_L_3body_to_p(self, L):
    p1, S0, L2 = svd(L)
    ## svds have bad precision

    p1 = p1[:,:self.D]
    S0 = S0[:self.D]
    L2 = L2[:self.D,:]

    S0 = np.diag(np.sqrt(S0))
    p1 = np.dot(p1, S0)
    L2 = np.dot(S0, L2)
    p1 = np.reshape(p1, (self.D, 2, 2, self.D))
    L2 = np.reshape(L2, (self.D*2*2, 2*2*self.D))
    p2, p3 = self.__decompose_L_2body_to_p(L2)

    return p1, p2, p3

  def operate_self(self, index):
    #operate 1-body interactions
    if(index < 0 or index >= self.N):
      raise NameError("Index %d out of range - operate_self" % index)
    L_self = copy.deepcopy(self.L_self)
    L_self = np.reshape(L_self, (4, 4))
    L_self = I4 + L_self*self.T_delta

    p_tmp = np.transpose(self.p_set[index], (1, 2, 0, 3))
    p_tmp = np.reshape(p_tmp, (2*2, self.D*self.D))
    p_tmp = np.reshape(np.dot(L_self, p_tmp), (2, 2, self.D, self.D))
    self.p_set[index] = np.transpose(p_tmp, (2, 0, 1, 3))

  def operate_2body(self, index1, index2):
    #operate 2-body interactions
    if(index1 < 0 or index1 >= self.N):
      raise NameError("Index %d out of range - operate_self" % index1)
    if(index2 < 0 or index2 >= self.N):
      raise NameError("Index %d out of range - operate_self" % index2)

    L_2body = copy.deepcopy(self.L_2body)
    L_2body = np.reshape(L_2body, (4*4, 4*4))

    p_tmp1 = np.reshape(self.p_set[index1], (self.D*2*2, self.D))
    p_tmp2 = np.reshape(self.p_set[index2], (self.D, 2*2*self.D))

    P = np.dot(p_tmp1, p_tmp2)
    P = np.reshape(P, (self.D, 2, 2, 2, 2, self.D))
    P = np.transpose(P, (1, 3, 2, 4, 0, 5))
    P = np.reshape(P, (2*2*2*2, self.D*self.D))

    L_2body = I16 + L_2body*self.T_delta
    L_2body = np.dot(L_2body, P)

    L_2body = np.reshape(L_2body, (2, 2, 2, 2, self.D, self.D))
    L_2body = np.transpose(L_2body, (4, 0, 2, 1, 3, 5))
    L_2body = np.reshape(L_2body, (self.D*2*2, 2*2*self.D))

    p_tmp1, p_tmp2 = self.__decompose_L_2body_to_p(L_2body)

    self.p_set[index1] = np.reshape(p_tmp1, (self.D, 2, 2, self.D))
    self.p_set[index2] = np.reshape(p_tmp2, (self.D, 2, 2, self.D))

  def operate_3body(self, index1, index2, index3):
    #operate 3-body interactions
    if(index1 < 0 or index1 >= self.N):
      raise NameError("Index %d out of range - operate_self" % index1)
    if(index2 < 0 or index2 >= self.N):
      raise NameError("Index %d out of range - operate_self" % index2)
    if(index3 < 0 or index3 >= self.N):
      raise NameError("Index %d out of range - operate_self" % index3)

    L_3body = copy.deepcopy(self.L_3body)
    L_3body = np.reshape(L_3body, (4*4*4, 4*4*4))

    p_tmp1 = np.reshape(self.p_set[index1], (self.D*2*2, self.D))
    p_tmp2 = np.reshape(self.p_set[index2], (self.D, 2*2*self.D))
    P = np.dot(p_tmp1, p_tmp2)
    P = np.reshape(P, (self.D*2*2*2*2, self.D))
    p_tmp3 = np.reshape(self.p_set[index3], (self.D, 2*2*self.D))
    P = np.dot(P, p_tmp3)
    P = np.reshape(P, (self.D, 2, 2, 2, 2, 2, 2, self.D))
    P = np.transpose(P, (1, 3, 5, 2, 4, 6, 0, 7))
    P = np.reshape(P, (2*2*2*2*2*2, self.D*self.D))

    L_3body = I64 + L_3body*self.T_delta
    L_3body = np.dot(L_3body, P)

    L_3body = np.reshape(L_3body, (2, 2, 2, 2, 2, 2, self.D, self.D))
    L_3body = np.transpose(L_3body, (6, 0, 3, 1, 4, 2, 5, 7))
    L_3body = np.reshape(L_3body, (self.D*2*2, 2*2*2*2*self.D))

    p_tmp1, p_tmp2, p_tmp3 = self.__decompose_L_3body_to_p(L_3body)

    self.p_set[index1] = np.reshape(p_tmp1, (self.D, 2, 2, self.D))
    self.p_set[index2] = np.reshape(p_tmp2, (self.D, 2, 2, self.D))
    self.p_set[index3] = np.reshape(p_tmp3, (self.D, 2, 2, self.D))

