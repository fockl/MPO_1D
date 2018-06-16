import numpy as np

Sz = np.array(([1, 0], [0, -1]), dtype=np.complex64)
Sx = np.array(([0, 1], [1, 0]), dtype=np.complex64)
Sy = np.array(([0, -1j], [1j, 0]), dtype=np.complex64)

I2 = np.identity(2, dtype=np.complex64)
I4 = np.identity(4, dtype=np.complex64)
I8 = np.identity(8, dtype=np.complex64)
I16 = np.identity(16, dtype=np.complex64)
I64 = np.identity(64, dtype=np.complex64)
