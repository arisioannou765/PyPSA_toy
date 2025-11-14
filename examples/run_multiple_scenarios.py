"""
Example usage of the ToyPyPSAModel OOP structure.

This script demonstrates:
- Creating a model with overridden parameters
- Multiple scnenario showcase
Run with:
    run_multiple_scenarios.py
"""
import os
import sys

# Ensure the project root (where toy_pypsa_model.py lives) is on sys.path
HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from toy_pypsa_model import ToyPyPSAModel

scenarios = [
    {"name": "base"},
    {"name": "cp_50", "carbon_price_eur_per_t": 50},
    {"name": "wind_push", "wind_p_nom_max": 25},
]

for s in scenarios:
    model = ToyPyPSAModel(**s)
    model.build_network()
    model.optimize()
    k = model.kpis.compute()
    print(model.name, k)
