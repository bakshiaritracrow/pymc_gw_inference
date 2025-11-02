import numpy as np
import os

N = 50  # Number of binaries
np.random.seed(42)

# Masses
m1 = np.random.uniform(5, 50, N)
m2 = np.random.uniform(5, m1)  # ensure m2 <= m1

# Spins: s1x, s1y, s1z, s2x, s2y, s2z
s1x = np.random.uniform(-0.99, 0.99, N)
s1y = np.random.uniform(-0.99, 0.99, N)
s1z = np.random.uniform(-0.99, 0.99, N)
s2x = np.random.uniform(-0.99, 0.99, N)
s2y = np.random.uniform(-0.99, 0.99, N)
s2z = np.random.uniform(-0.99, 0.99, N)

# Chirp mass
chirp_mass = ((m1 * m2)**(3/5)) / ((m1 + m2)**(1/5))

# Chi effective
chi_eff = (s1z * m1 + s2z * m2) / (m1 + m2)

os.makedirs("data", exist_ok=True)
np.savez('data/mock_binary_data.npz', m1=m1, m2=m2, s1x=s1x, s1y=s1y, s1z=s1z,
         s2x=s2x, s2y=s2y, s2z=s2z, chirp_mass=chirp_mass, chi_eff=chi_eff)
print("Mock binary data saved to data/mock_binary_data.npz")
