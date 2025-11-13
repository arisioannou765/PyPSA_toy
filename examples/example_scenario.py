"""
Example usage of the ToyPyPSAModel OOP structure.

This script demonstrates:
- Creating a model with overridden parameters
- Building and optimizing the PyPSA network
- Running sanity checks
- Computing KPIs
- Proucing plots

Run with:
    python examples/example_scenario.py
"""

from toy_pypsa_model import ToyPyPSAModel

def main():

    # Create a model with a nonzero carbon price
    model = ToyPyPSAModel(name="example_scenario", carbon_price_eur_per_t=50)

    print("\n=== Building network ===")
    model.build_network()

    print("\n=== Optimizing ===")
    model.optimize()

    print("\n=== Sanity checks ===")
    checks = model.sanity_checks()
    print(checks)

    print("\n=== KPIs ===")
    k = model.kpis.compute()
    for key, val in k.items():
        print(f"{key}: {val}")

    print("\n=== Plotting ===")
    model.plots.dispatch_stack(outfile="dispatch_example.png")
    model.plots.emissions_by_generator_stack(outfile="emissions_example.png")

    print("\nExample scenario completed.")

if __name__ == "__main__":
    main()
