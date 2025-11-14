#Testing ground

from toy_pypsa_model import ToyPyPSAModel


def main():
    # create a model with an overridden name and carbon price
    model = ToyPyPSAModel(name="carbon_50", carbon_price_eur_per_t=50)

    print("Model name:", model.name)
    print("Carbon price:", model.params["carbon_price_eur_per_t"])
    print("Hours:", model.params["hours"])
    print("Demand base:", model.params["demand_base_MW"])

    # build the network and capture the returned object
    n = model.build_network()

    print("  Network name:", n.name)
    print("  Number of snapshots:", len(n.snapshots))
    print("\nGenerators table:")
    print(n.generators[["carrier", "p_nom_extendable"]])

    #Optimize the network
    print("\nOptimizing network...")
    model.optimize()  # this will use self.network (already built)

    # After optimization, PyPSA should have added p_nom_opt etc.
    print("\nAfter optimization:")
    if "p_nom_opt" in n.generators.columns:
        print("  Optimal capacities (MW):")
        print(n.generators[["carrier", "p_nom_opt"]])
    else:
        print("  Warning: 'p_nom_opt' not found in generators table.")

    #check sanity checks method
    checks = model.sanity_checks()
    print("\nSanity checks dict:")
    print(checks)

    # KPI dictionary
    print("\ Computing all KPIs ")
    kpi_dict = model.kpis.compute()
    print(kpi_dict)

    # Individual KPIs
    print("\n Individual KPI methods")
    print("Total emissions:", model.kpis.emissions_total())
    print("Unserved energy:", model.kpis.unserved_energy())

    caps, gens_used, cap_col = model.kpis.capacity_mix_and_generators()
    print("Capacity mix:", caps)
    print("Generators used:", gens_used)
    print("Capacity column used:", cap_col)

    print("Mean price:", model.kpis.mean_price())
    print("Curtailment by carrier:", model.kpis.curtailment_by_carrier(cap_col))

    #Plotting test
    print("\n Plotting dispatch stack")
    df_dispatch = model.plots.dispatch_stack(outfile="dispatch_stack.png")
    print("Dispatch dataframe columns:", df_dispatch.columns.tolist())

    print("\n Plotting emissions by generator")
    emis_data = model.plots.emissions_by_generator_stack(
        outfile="emissions_by_generator_stack.png"
    )
    print("Emissions data keys:", list(emis_data.keys()))


if __name__ == "__main__":
    main()
