import numpy as np
from scipy.linalg import svd
#from scipy.sparse.linalg import svds
import copy
from Physical_operators import *

class MPO_1D:
  def __init__(self, N, D, T_delta, H_1body_kind=1, K_1body_kind=1, H_2body_kind=1, H_3body_kind=1):
    self.T_delta = T_delta
    self.N = N
    self.D = D

    if(K_1body_kind!=1):
      raise NameError("K_1body_kind must be 1 for now")

    self.left_edge_vector = np.ones(self.D, dtype=np.complex64)/np.sqrt(self.D)
    self.right_edge_vector = np.ones(self.D, dtype=np.complex64)/np.sqrt(self.D)

    self.H_1body_kind = H_1body_kind
    self.K_1body_kind = K_1body_kind
    self.H_2body_kind = H_2body_kind
    self.H_3body_kind = H_3body_kind

    self.H_1body = np.zeros(((H_1body_kind, 2, 2)), dtype=np.complex64)
    self.K_1body = np.zeros(((K_1body_kind, 2, 2)), dtype=np.complex64)
    self.H_2body = np.zeros(((H_2body_kind, 4, 4)), dtype=np.complex64)
    self.H_3body = np.zeros(((H_3body_kind, 8, 8)), dtype=np.complex64)
    self.L_self = np.zeros(((((H_1body_kind, 2, 2, 2, 2)))), dtype=np.complex64)
    self.L_2body = np.zeros(((((H_2body_kind, 4, 4, 4, 4)))), dtype=np.complex64)
    self.L_3body = np.zeros(((((H_3body_kind, 8, 8, 8, 8)))), dtype=np.complex64)

    self.p_set = np.zeros(((((self.N, self.D, 2, 2, self.D)))), dtype=np.complex64)

  def __Liouvillian_self(self, index=-1):
    if(index<0):
      for loop_index in range(self.H_1body_kind):
        self.L_self[loop_index] = np.zeros((((2, 2, 2, 2))), dtype=np.complex64)
        H_transpose = np.transpose(self.H_1body[loop_index])
        K_transpose = np.transpose(self.K_1body[0])
        K_conjugate = np.conjugate(self.K_1body[0])
        K_dagger = np.conjugate(K_transpose)
        K1 = np.dot(K_dagger, self.K_1body[0])
        K2 = np.dot(K_transpose, K_conjugate)

        for i in range(2):
          for j in range(2):
            for k in range(2):
              for l in range(2):
                self.L_self[loop_index][i][j][k][l] = -1j*(self.H_1body[loop_index][i][k]*I2[j][l] - I2[i][k]*H_transpose[j][l])
                self.L_self[loop_index][i][j][k][l] = self.L_self[loop_index][i][j][k][l] + self.K_1body[0][i][k]*K_conjugate[j][l]
                self.L_self[loop_index][i][j][k][l] = self.L_self[loop_index][i][j][k][l] - 0.5*K1[i][k]*I2[j][l] - 0.5*I2[i][k]*K2[j][l]
    else:
      if(index >= self.H_1body_kind):
        raise NameError("__Liouvillian_self : index %d must be smaller than %d" % (index, self.H_1body_kind))
      self.L_self[index] = np.zeros((((2, 2, 2, 2))), dtype=np.complex64)
      H_transpose = np.transpose(self.H_1body[index])
      K_transpose = np.transpose(self.K_1body[0])
      K_conjugate = np.conjugate(self.K_1body[0])
      K_dagger = np.conjugate(K_transpose)
      K1 = np.dot(K_dagger, self.K_1body[0])
      K2 = np.dot(K_conjugate, K_transpose)

      for i in range(2):
        for j in range(2):
          for k in range(2):
            for l in range(2):
              self.L_self[index][i][j][k][l] = -1j*(self.H_1body[index][i][k]*I2[j][l] - I2[i][k]*H_transpose[j][l])
              self.L_self[index][i][j][k][l] = self.L_self[index][i][j][k][l] + self.K_1body[0][i][k]*K_conjugate[j][l]
              self.L_self[index][i][j][k][l] = self.L_self[index][i][j][k][l] - 0.5*K1[i][k]*I2[j][l] - 0.5*I2[i][k]*K2[j][l]

  def __Liouvillian_2body(self, index=-1):
    if(index<0):
      for loop_index in range(self.H_2body_kind):
        self.L_2body[loop_index] = np.zeros((((4, 4, 4, 4))), dtype=np.complex64)
        H_transpose = np.transpose(self.H_2body[loop_index])

        for i in range(4):
          for j in range(4):
            for k in range(4):
              for l in range(4):
                self.L_2body[loop_index][i][j][k][l] = -1j*(self.H_2body[loop_index][i][k]*I4[j][l] - I4[i][k]*H_transpose[j][l])
    else:
      if(index >= self.H_2body_kind):
        raise NameError("__Liouvillian_2body : index %d must be smaller than %d" % (index, self.H_2body_kind))
      self.L_2body[index] = np.zeros((((4, 4, 4, 4))), dtype=np.complex64)
      H_transpose = np.transpose(self.H_2body[index])

      for i in range(4):
        for j in range(4):
          for k in range(4):
            for l in range(4):
              self.L_2body[index][i][j][k][l] = -1j*(self.H_2body[index][i][k]*I4[j][l] - I4[i][k]*H_transpose[j][l])

  def __Liouvillian_3body(self, index=-1):
    if(index<0):
      for loop_index in range(self.H_3body_kind):
        self.L_3body[loop_index] = np.zeros((((8, 8, 8, 8))), dtype=np.complex64)
        H_transpose = np.transpose(self.H_3body[loop_index])

        for i in range(8):
          for j in range(8):
            for k in range(8):
              for l in range(8):
                self.L_3body[loop_index][i][j][k][l] = -1j*(self.H_3body[loop_index][i][k]*I8[j][l] - I8[i][k]*H_transpose[j][l])
    else:
      if(index >= self.H_3body_kind):
        raise NameError("__Liouvillian_3body : index %d must be smaller than %d" % (index, self.H_3body_kind))
      self.L_3body[index] = np.zeros((((8, 8, 8, 8))), dtype=np.complex64)
      H_transpose = np.transpose(self.H_3body[index])

      for i in range(8):
        for j in range(8):
          for k in range(8):
            for l in range(8):
              self.L_3body[index][i][j][k][l] = -1j*(self.H_3body[index][i][k]*I8[j][l] - I8[i][k]*H_transpose[j][l])

  def initialize_H_1body(self, index=-1):
    if(index<0):
      for loop_index in range(self.H_1body_kind):
        self.H_1body[loop_index] = np.zeros((2, 2), dtype=np.complex64)
    else:
      if(index>=self.H_1body_kind):
        raise NameError("initialize_H_1body : index %d must be smaller than H_1body_kind %d" % (index, self.H_1body_kind))
      self.H_1body[index] = np.zeros((2, 2), dtype=np.complex64)
    self.__Liouvillian_self(index)

  def add_H_1body(self, H_1body, coef=1.0, index=0):
    if(index<0 or index>=self.H_1body_kind):
      raise NameError("add_H_1body : index %d must be smaller than H_1body_kind %d" % (index, self.H_1body_kind))
    if(H_1body.shape != (2, 2)):
      raise NameError("H_1body type should be (2, 2)")
    H_tmp = np.zeros((2, 2), dtype=np.complex64)
    for i0 in range(2):
      for i1 in range(2):
        H_tmp[i0][i1] = H_1body[i0][i1]
    self.H_1body[index] = self.H_1body[index] + coef*H_tmp
    self.__Liouvillian_self(index)

  def initialize_K_1body(self, index=-1):
    if(index<0):
      for loop_index in range(self.K_1body_kind):
        self.K_1body[loop_index] = np.zeros((2, 2), dtype=np.complex64)
    else:
      if(index>=self.K_1body_kind):
        raise NameError("initialize_K_1body : index %d must be smaller than K_1body_kind %d" % (index, self.K_1body_kind))
      self.K_1body[index] = np.zeros((2, 2), dtype=np.complex64)
    self.__Liouvillian_self(index)

  def add_K_1body(self, K_1body, coef=1.0, index=0):
    if(index<0 or index>=self.K_1body_kind):
      raise NameError("add_K_1body : index %d must be smaller than K_1body_kind %d" % (index, self.K_1body_kind))
    if(K_1body.shape != (2, 2)):
      raise NameError("K_1body type should be (2, 2)")
    K_tmp = np.zeros((2, 2), dtype=np.complex64)
    for i0 in range(2):
      for i1 in range(2):
        K_tmp[i0][i1] = K_1body[i0][i1]
    self.K_1body[index] = self.K_1body[index]+ coef*K_tmp
    self.__Liouvillian_self()

  def initialize_H_2body(self, index=-1):
    if(index<0):
      for loop_index in range(self.H_2body_kind):
        self.H_2body[loop_index] = np.zeros((4, 4), dtype=np.complex64)
    else:
      if(index>=self.H_2body_kind):
        raise NameError("initialize_H_2body : index %d must be smaller than H_2body_kind" % (index, self.H_2body_kind))
      self.H_2body[index] = np.zeros((4, 4), dtype=np.complex64)
    self.__Liouvillian_2body(index)

  def add_H_2body(self, H_2body1, H_2body2, coef=1.0, index=0):
    if(index<0 or index>=self.H_2body_kind):
      raise NameError("add_H_2body : index %d must be smaller than H_2body_kind %d" % (index, self.H_2body_kind))
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
    self.H_2body[index] = self.H_2body[index] + coef*H_tmp
    self.__Liouvillian_2body()

  def initialize_H_3body(self, index=-1):
    if(index<0):
      for loop_index in range(self.H_3body_kind):
        self.H_3body[loop_index] = np.zeros((8, 8), dtype=np.complex64)
    else:
      if(index>=self.H_3body_kind):
        raise NameError("initialize_H_3body : index %d must be smaller than H_3body_kind" % (index, self.H_3body_kind))
      self.H_3body[index] = np.zeros((8, 8), dtype=np.complex64)
    self.__Liouvillian_3body(index)

  def add_H_3body(self, H_3body1, H_3body2, H_3body3, coef=1.0, index=0):
    if(index<0 or index>=self.H_3body_kind):
      raise NameError("add_H_3body : index %d must be smaller than H_3body_kind %d" % (index, self.H_3body_kind))
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
    self.H_3body[index] = self.H_3body[index] + coef*H_tmp
    self.__Liouvillian_3body(index)

  def trace_of_density_matrix(self):
    v_set = np.zeros(((self.N, self.D, self.D)), dtype=np.complex64)
    for p_num in range(self.N):
      for D1 in range(self.D):
        for D2 in range(self.D):
          for i in range(2):
            v_set[p_num][D1][D2] = v_set[p_num][D1][D2] + self.p_set[p_num][D1][i][i][D2]

    v_left  = copy.deepcopy(self.left_edge_vector)
    #product_v = np.zeros((self.D, self.D), dtype=np.complex64)
    for p_num in range(self.N):
      v_left = np.dot(v_left, v_set[p_num])
    #for p_num in range(self.N-1):
    #  product_v = np.dot(v_set[p_num], v_set[p_num+1])
    #  v_set[p_num+1] = product_v

    return np.dot(v_left, self.left_edge_vector).real
    #return np.trace(product_v).real

  def normalize(self):
    leftover = np.identity(self.D, dtype=np.complex64)

    for n in range(self.N):
      p_set_tmp = np.dot(leftover, np.reshape(self.p_set[n], (self.D, 2*2*self.D)))
      p_set_tmp = np.reshape(p_set_tmp, (self.D*2*2, self.D))
      U, S, V = svd(p_set_tmp, full_matrices=False)
      self.p_set[n] = np.reshape(U, (self.D, 2, 2, self.D))
      leftover = np.dot(np.diag(S), V)

    for n_tmp in range(self.N-1):
      n = self.N-1 - n_tmp
      p_set_tmp = np.dot(np.reshape(self.p_set[n], (self.D*2*2, self.D)), leftover)
      p_set_tmp = np.reshape(p_set_tmp, (self.D, 2*2*self.D))
      U, S, V = svd(p_set_tmp, full_matrices=False)
      S1 = np.diag(np.power(S, n/(n+1.0)))
      S2 = np.diag(np.power(S, 1.0/(n+1.0)))
      self.p_set[n] = np.reshape(np.dot(S2, V), (self.D, 2, 2, self.D))
      leftover = np.dot(U, S1)

    p_set_tmp = np.dot(np.reshape(self.p_set[0], (self.D*2*2, self.D)), leftover)
    self.p_set[0] = np.reshape(p_set_tmp, (self.D, 2, 2, self.D))

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

  def calculate_expectation_of_1body(self, O=Sz, index=-1):
    if(index >= self.N):
      raise NameError("calculate_expectation_of_1body : index %d out of range" % (index))
    if(O.shape != (2, 2)):
      raise NameError("operator shape should be (2, 2)")
    
    p_set_0 = np.zeros(((self.N, self.D, self.D)), dtype=np.complex64)
    p_set_1 = np.zeros(((self.N, self.D, self.D)), dtype=np.complex64)
    for n in range(self.N):
      for D1 in range(self.D):
        for D2 in range(self.D):
          for d in range(2):
            p_set_0[n][D1][D2] = p_set_0[n][D1][D2] + self.p_set[n][D1][d][d][D2]

      p_tmp = np.transpose(self.p_set[n], (1, 0, 2, 3))
      p_tmp = np.reshape(p_tmp, (2, (self.D*2*self.D)))
      p_tmp = np.dot(O, p_tmp)
      p_tmp = np.reshape(p_tmp, (2, self.D, 2, self.D))
      p_tmp = np.transpose(p_tmp, (1, 0, 2, 3))

      for D1 in range(self.D):
        for D2 in range(self.D):
          for d in range(2):
            p_set_1[n][D1][D2] = p_set_1[n][D1][D2] + p_tmp[D1][d][d][D2]

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

    calc_O = np.zeros(self.N, dtype=np.complex64)

    calc_O[0] = np.dot(np.dot(v_left, p_set_1[0]), p_product_right[1])
    #calc_O[0] = np.trace(np.dot(p_set_1[0], p_product_right[1]))
    for n in range(1, self.N-1):
      calc_O[n] = np.dot(p_product_left[n-1], np.dot(p_set_1[n], p_product_right[n+1]))
      #calc_O[n] = np.trace(np.dot(p_product_left[n-1], np.dot(p_set_1[n], p_product_right[n+1])))
    calc_O[self.N-1] = np.dot(p_product_left[self.N-2], np.dot(p_set_1[self.N-1], v_right))
    #calc_O[self.N-1] = np.trace(np.dot(p_product_left[self.N-2], p_set_1[self.N-1]))

    if(index<0):
      return calc_O
    else:
      return calc_O[index]

  def calculate_expectation_of_2body(self, O1=Sz, O2=Sz, dist=1, index=-1):
    if(index+dist >= self.N):
      raise NameError("calculate_expectation_of_2body : index %d with dist %d out of range" % (index, dist))
    elif(dist < 1):
      raise NameError("calculate_expectation_of_2body : dist %d must be larger than 0" % (dist))
    if(O1.shape != (2, 2)):
      raise NameError("operator O1 should be (2, 2)")
    if(O2.shape != (2, 2)):
      raise NameError("operator O2 should be (2, 2)")
 
    p_set_0 = np.zeros(((self.N, self.D, self.D)), dtype=np.complex64)
    p_set_1 = np.zeros(((self.N, self.D, self.D)), dtype=np.complex64)
    p_set_2 = np.zeros(((self.N, self.D, self.D)), dtype=np.complex64)
    for n in range(self.N):
      for D1 in range(self.D):
        for D2 in range(self.D):
          for d in range(2):
            p_set_0[n][D1][D2] = p_set_0[n][D1][D2] + self.p_set[n][D1][d][d][D2]

      p_tmp = np.transpose(self.p_set[n], (1, 0, 2, 3))
      p_tmp = np.reshape(p_tmp, (2, (self.D*2*self.D)))
      p_tmp = np.dot(O1, p_tmp)
      p_tmp = np.reshape(p_tmp, (2, self.D, 2, self.D))
      p_tmp = np.transpose(p_tmp, (1, 0, 2, 3))

      for D1 in range(self.D):
        for D2 in range(self.D):
          for d in range(2):
            p_set_1[n][D1][D2] = p_set_1[n][D1][D2] + p_tmp[D1][d][d][D2]

      p_tmp = np.transpose(self.p_set[n], (1, 0, 2, 3))
      p_tmp = np.reshape(p_tmp, (2, (self.D*2*self.D)))
      p_tmp = np.dot(O2, p_tmp)
      p_tmp = np.reshape(p_tmp, (2, self.D, 2, self.D))
      p_tmp = np.transpose(p_tmp, (1, 0, 2, 3))

      for D1 in range(self.D):
        for D2 in range(self.D):
          for d in range(2):
            p_set_2[n][D1][D2] = p_set_2[n][D1][D2] + p_tmp[D1][d][d][D2]

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

    calc_O_O = np.zeros(self.N-dist, dtype=np.complex64)

    calc_tmp = np.dot(v_left, p_set_1[0])
    for i in range(1, dist):
      calc_tmp = np.dot(calc_tmp, p_set_0[i])
    calc_tmp = np.dot(calc_tmp, p_set_2[dist])
    if(dist+1 < self.N):
      calc_O_O[0] = np.dot(calc_tmp, p_product_right[dist+1])
    elif(dist+1 == self.N):
      calc_O_O[0] = np.dot(calc_tmp, v_right)
    else:
      raise NameError("calculate_SzSz_correlation : dist+1 = %d > self.N = %d" % (dist+1, self.N))

    for n in range(1, self.N-1-dist):
      calc_tmp = np.dot(p_product_left[n-1], p_set_1[n])
      for i in range(1, dist):
        calc_tmp = np.dot(calc_tmp, p_set_0[n+i])
      calc_tmp = np.dot(calc_tmp, p_set_2[n+dist])
      calc_O_O[n] = np.dot(calc_tmp, p_product_right[n+dist+1])

    if(self.N-2-dist >= 0):
      calc_tmp = np.dot(p_product_left[self.N-2-dist], p_set_1[self.N-1-dist])
    elif(self.N-2-dist == -1):
      calc_tmp = np.dot(v_left, p_set_1[self.N-1-dist])
    else:
      raise NameError("calculate_SzSz_correlation : self.N-2-dist = %d >= -1" % (self.N-2-dist))
    for i in range(1, dist):
      calc_tmp = np.dot(calc_tmp, p_set_0[i+self.N-1-dist])
    calc_tmp = np.dot(calc_tmp, p_set_2[self.N-1])
    calc_O_O[self.N-1-dist] = np.dot(calc_tmp, v_right)

    if(index<0):
      return calc_O_O
    else:
      return calc_O_O[index]

  def calculate_entanglement_entropy(self, index1, index2):
    if(index1<0 or index1>=self.N):
      raise NameError("calculate_entanglement_entropy : index1 out of range")
    if(index2<0 or index2>=self.N):
      raise NameError("calculate_entanglement_entropy : index2 out of range")
    if(index1>index2):
      raise NameError("calculate_entanglement_entropy : index2 %d must be larger than or equal to index 1 %d" % (index2, index1))

    v_left  = copy.deepcopy(self.left_edge_vector)
    v_right = copy.deepcopy(self.right_edge_vector)

    for n in range(index1):
      p_tmp = np.zeros((self.D, self.D), dtype=np.complex64)
      for D1 in range(self.D):
        for D2 in range(self.D):
          for d in range(2):
            p_tmp[D1][D2] = p_tmp[D1][D2] + self.p_set[n][D1][d][d][D2]

      v_left = np.dot(v_left, p_tmp)

    for n in range(self.N-1-index2):
      p_tmp = np.zeros((self.D, self.D), dtype=np.complex64)
      for D1 in range(self.D):
        for D2 in range(self.D):
          for d in range(2):
            p_tmp[D1][D2] = p_tmp[D1][D2] + self.p_set[self.N-1-n][D1][d][d][D2]
      v_right = np.dot(p_tmp, v_right)

    p_tmp = copy.deepcopy(self.p_set[index1])
    p_tmp = np.reshape(p_tmp, (self.D, 2*2*self.D))
    p_local = np.dot(v_left, p_tmp)
    num_of_comp = 2
    p_local = np.reshape(p_local, (num_of_comp*num_of_comp, self.D))
    for n in range(index1+1, index2+1):
      p_tmp = copy.deepcopy(self.p_set[n])
      p_tmp = np.reshape(p_tmp, (self.D, 2*2*self.D))
      p_local = np.dot(p_local, p_tmp)
      p_local = np.reshape(p_local, (num_of_comp, num_of_comp, 2, 2, self.D))
      num_of_comp = num_of_comp * 2
      p_local = np.transpose(p_local, (0, 2, 1, 3, 4))
      p_local = np.reshape(p_local, (num_of_comp*num_of_comp, self.D))

    p_local = np.dot(p_local, v_right)
    p_local = np.reshape(p_local, (num_of_comp, num_of_comp))

    S = np.linalg.eigvalsh(p_local)
    ans = 0.0

    for i in range(len(S)):
      ans = ans + S[i]*np.log(S[i])

    return -ans

  def __decompose_L_2body_to_p(self, L):
    p1, S0, p2 = svd(L, full_matrices=False)
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
    p1, S0, L2 = svd(L, full_matrices=False)
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

  def operate_1body(self, particle_index, Liouvillian_index=0):
    #operate 1-body interactions
    if(particle_index < 0 or particle_index >= self.N):
      raise NameError("particle_index %d out of range - operate_1body" % (particle_index))
    if(Liouvillian_index < 0 or Liouvillian_index >= self.H_1body_kind):
      raise NameError("Liouvillian_index %d out of range - operate_1body" % (Liouvillian_index))
    L_self = copy.deepcopy(self.L_self[Liouvillian_index])
    L_self = np.reshape(L_self, (4, 4))
    L_self = I4 + L_self*self.T_delta

    p_tmp = np.transpose(self.p_set[particle_index], (1, 2, 0, 3))
    p_tmp = np.reshape(p_tmp, (2*2, self.D*self.D))
    p_tmp = np.reshape(np.dot(L_self, p_tmp), (2, 2, self.D, self.D))
    self.p_set[particle_index] = np.transpose(p_tmp, (2, 0, 1, 3))

  def operate_2body(self, particle_index1, particle_index2, Liouvillian_index=0):
    #operate 2-body interactions
    if(particle_index1 < 0 or particle_index1 >= self.N):
      raise NameError("particle_index %d out of range - operate_2body" % (particle_index1))
    if(particle_index2 < 0 or particle_index2 >= self.N):
      raise NameError("particle_index %d out of range - operate_2body" % (particle_index2))
    if(Liouvillian_index < 0 or Liouvillian_index >= self.H_2body_kind):
      raise NameError("Liouvillian_index %d out of range - operate_2body" % (Liouvillian_index))

    if(particle_index1+1 != particle_index2):
      raise NameError("particle_index1 + 1 (%d+1) must be particle_index2 (%d)" % (particle_index1, particle_index2))

    L_2body = copy.deepcopy(self.L_2body[Liouvillian_index])
    L_2body = np.reshape(L_2body, (4*4, 4*4))

    p_tmp1 = np.reshape(self.p_set[particle_index1], (self.D*2*2, self.D))
    p_tmp2 = np.reshape(self.p_set[particle_index2], (self.D, 2*2*self.D))

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

    self.p_set[particle_index1] = np.reshape(p_tmp1, (self.D, 2, 2, self.D))
    self.p_set[particle_index2] = np.reshape(p_tmp2, (self.D, 2, 2, self.D))

  def operate_3body(self, particle_index1, particle_index2, particle_index3, Liouvillian_index=0):
    #operate 3-body interactions
    if(particle_index1 < 0 or particle_index1 >= self.N):
      raise NameError("particle_index %d out of range - operate_3body" % particle_index1)
    if(particle_index2 < 0 or particle_index2 >= self.N):
      raise NameError("particle_index %d out of range - operate_3body" % particle_index2)
    if(particle_index3 < 0 or particle_index3 >= self.N):
      raise NameError("particle_index %d out of range - operate_3body" % particle_index3)
    if(Liouvillian_index < 0 or Liouvillian_index >= self.H_3body_kind):
      raise NameError("Liouvillian_index %d out of range - operate_3body" % (Liouvillian_index))
    if(particle_index1+1 != particle_index2):
      raise NameError("particle_index1+1 (%d+1) must be particle_index2 (%d)" % (particle_index1, particle_index2))
    if(particle_index2+1 != particle_index3):
      raise NameError("particle_index2+1 (%d+1) must be particle_index3 (%d)" % (particle_index2, particle_index3))

    L_3body = copy.deepcopy(self.L_3body[Liouvillian_index])
    L_3body = np.reshape(L_3body, (4*4*4, 4*4*4))

    p_tmp1 = np.reshape(self.p_set[particle_index1], (self.D*2*2, self.D))
    p_tmp2 = np.reshape(self.p_set[particle_index2], (self.D, 2*2*self.D))
    P = np.dot(p_tmp1, p_tmp2)
    P = np.reshape(P, (self.D*2*2*2*2, self.D))
    p_tmp3 = np.reshape(self.p_set[particle_index3], (self.D, 2*2*self.D))
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

    self.p_set[particle_index1] = np.reshape(p_tmp1, (self.D, 2, 2, self.D))
    self.p_set[particle_index2] = np.reshape(p_tmp2, (self.D, 2, 2, self.D))
    self.p_set[particle_index3] = np.reshape(p_tmp3, (self.D, 2, 2, self.D))

