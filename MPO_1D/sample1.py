import numpy as np
from Physical_operators import *
from MPO_1D import MPO_1D
#exact Sz for check
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

#exact Sz for check
def exact_SzSz_correlation_average(t, calculate_index, h, J, V, dist=1):
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

if __name__ == '__main__':
  N=3
  D=4
  J=-1.0
  h=1.0
  V=-1.0
  T_init=-10
  T_end=0 + 1.0E-3
  T_delta=1.0E-3

  TEST = MPO_1D(N=N, D=D, T_delta=T_delta, H_1body_kind=3)

  print("initialize density matrix")
  TEST.initialize_density_matrix()

  print("initial <Sz>")
  print(TEST.calculate_expectation_of_1body(Sz))

  filename = "sample1.out"
  f = open(filename, "w")
  f.close()

  TEST.initialize_H_1body()
  TEST.add_H_1body(Sx, h, index=0)
  TEST.add_H_1body(Sx, h, index=1)
  TEST.add_H_1body(Sx, h, index=2)
  TEST.initialize_K_1body()

  TEST.initialize_H_2body()
  TEST.add_H_2body(Sz, Sz, -J)
  TEST.initialize_H_3body()
  TEST.add_H_3body(Sz, I, Sz, -V)

  t = T_init
  while(t < T_end):
    #operate 1-body interactions
    for n in range(N):
      TEST.operate_1body(n, Liouvillian_index=n)

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

    Sz_ave_array = TEST.calculate_expectation_of_1body(O=Sz)
    Sz_cor1_array = TEST.calculate_expectation_of_2body(O1=Sz, O2=Sz, dist=1)
    Sz_cor2_array = TEST.calculate_expectation_of_2body(O1=Sz, O2=Sz, dist=2)
    Sz_ave = sum(Sz_ave_array).real/len(Sz_ave_array)
    SzSz1_ave = sum(Sz_cor1_array).real/len(Sz_cor1_array)
    SzSz2_ave = sum(Sz_cor2_array).real/len(Sz_cor2_array)
    Sz_exact = exact_Sz_average(t-T_init, 1, h, J, V)
    SzSz1_exact = exact_SzSz_correlation_average(t-T_init, 1, h, J, V, dist=1)
    SzSz2_exact = exact_SzSz_correlation_average(t-T_init, 1, h, J, V, dist=2)

    Entanglement_Entropy = TEST.calculate_entanglement_entropy(1, 1).real

    f = open(filename, "a")
    print("%lf %lf %lf %lf %lf %lf %lf %lf\n" % (t, Sz_ave, Sz_exact, SzSz1_ave, SzSz1_exact, SzSz2_ave, SzSz2_exact, Entanglement_Entropy))
    f.write("%lf %lf %lf %lf %lf %lf %lf %lf\n" % (t, Sz_ave, Sz_exact, SzSz1_ave, SzSz1_exact, SzSz2_ave, SzSz2_exact, Entanglement_Entropy))
    f.close()

    if(np.abs(Sz_ave) > 2.0):
      break

    t = t + T_delta

