import pymc as pm
import numpy as np

def load_data(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}

def build_model(data):
    N = len(data['m1'])
    with pm.Model() as model:
        # Hyperpriors for population means
        mu_m1 = pm.Uniform('mu_m1', 5, 50)
        mu_m2 = pm.Uniform('mu_m2', 5, 50)
        mu_chi_eff = pm.Uniform('mu_chi_eff', -1, 1)

        # Population spreads
        sigma_m1 = pm.HalfNormal('sigma_m1', 10)
        sigma_m2 = pm.HalfNormal('sigma_m2', 10)
        sigma_chi_eff = pm.HalfNormal('sigma_chi_eff', 0.3)

        # Individual binary parameters (hierarchical model)
        m1 = pm.Normal('m1', mu=mu_m1, sigma=sigma_m1, shape=N)
        m2 = pm.Normal('m2', mu=mu_m2, sigma=sigma_m2, shape=N)

        # Spins
        s1x = pm.Uniform('s1x', -0.99, 0.99, shape=N)
        s1y = pm.Uniform('s1y', -0.99, 0.99, shape=N)
        s1z = pm.Uniform('s1z', -0.99, 0.99, shape=N)
        s2x = pm.Uniform('s2x', -0.99, 0.99, shape=N)
        s2y = pm.Uniform('s2y', -0.99, 0.99, shape=N)
        s2z = pm.Uniform('s2z', -0.99, 0.99, shape=N)

        # Chirp mass (deterministic)
        chirp_mass = pm.Deterministic('chirp_mass', ((m1*m2)**(3/5))/((m1+m2)**(1/5)))

        # Chi effective (deterministic)
        chi_eff = pm.Deterministic('chi_eff', (s1z*m1 + s2z*m2)/(m1 + m2))

        # Likelihoods
        obs_m1 = pm.Normal('obs_m1', mu=m1, sigma=0.5, observed=data['m1'])
        obs_m2 = pm.Normal('obs_m2', mu=m2, sigma=0.5, observed=data['m2'])
        obs_s1x = pm.Normal('obs_s1x', mu=s1x, sigma=0.2, observed=data['s1x'])
        obs_s1y = pm.Normal('obs_s1y', mu=s1y, sigma=0.2, observed=data['s1y'])
        obs_s1z = pm.Normal('obs_s1z', mu=s1z, sigma=0.2, observed=data['s1z'])
        obs_s2x = pm.Normal('obs_s2x', mu=s2x, sigma=0.2, observed=data['s2x'])
        obs_s2y = pm.Normal('obs_s2y', mu=s2y, sigma=0.2, observed=data['s2y'])
        obs_s2z = pm.Normal('obs_s2z', mu=s2z, sigma=0.2, observed=data['s2z'])
        obs_chirp_mass = pm.Normal('obs_chirp_mass', mu=chirp_mass, sigma=1.0, observed=data['chirp_mass'])
        obs_chi_eff = pm.Normal('obs_chi_eff', mu=chi_eff, sigma=sigma_chi_eff, observed=data['chi_eff'])

    return model
