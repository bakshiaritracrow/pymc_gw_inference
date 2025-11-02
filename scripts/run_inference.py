import pymc as pm
from models.binary_system_model import load_data, build_model
import arviz as az
import os

data = load_data('data/mock_binary_data.npz')
model = build_model(data)

os.makedirs("results", exist_ok=True)
with model:
    trace = pm.sample(2000, tune=1000, target_accept=0.95)
    az.to_netcdf(trace, 'results/binary_system_trace.nc')
    print("Inference complete. Results saved to results/binary_system_trace.nc")
