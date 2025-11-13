PyPSA Toy Model

This project contains an object-oriented implementation of a small PyPSA energy-system model.
It is designed to function as an extendable framework for grid modelling and scenario development.

The model is structured as a mini-library with separation of concerns:

ToyPyPSAModel: handles configuration, network construction, and optimization

KPIs: computes scenario metrics (eg. emissions, curtailment, cost, capacity mix, etc.)

Plots: visualizes output

test_toy_model.py: demonstrates the full end-to-end workflow

Features

Fully parameterized 24-hour representative-day model

Extendable set of carriers (wind, gas, coal, hydro, battery, load shedding)

Carbon pricing and capacity caps

Object-oriented structure:

model.build_network()

model.optimize()

model.sanity_checks()

model.kpis.compute() and individual KPI methods

model.plots.dispatch_stack() and emissions plots

Simple test script demonstrating a complete scenario

Project Structure

PyPSA_toy/
│
├── toy_pypsa_model.py        # Model, KPIs, and Plots classes
├── test_toy_model.py         # Demonstrates full model workflow
├── notebooks/                # Exploratory notebooks
│   └── PyPSA_toy.ipynb
├── examples/  
└── README.md

Installation

git clone https://github.com/arisioannou765/PyPSA_toy.git
cd PyPSA_toy

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Example Usage
from toy_pypsa_model import ToyPyPSAModel

# Create a model with custom parameters
model = ToyPyPSAModel(carbon_price_eur_per_t=50)

# Build and optimize
model.build_network()
model.optimize()

# Run sanity checks
model.sanity_checks()

# KPIs (dictionary output)
results = model.kpis.compute()
print(results)

# Plots
model.plots.dispatch_stack()
model.plots.emissions_by_generator_stack()

Run python examples/example_scenario.py for a full runnable example.
