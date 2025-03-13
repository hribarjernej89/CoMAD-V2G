# CoMAD-V2G: Community Multi-Agent Deep reinforcement learning Vehicle-to-Grid 🚗⚡️

CoMAD is a Multi-agent Reinforcement Learning (MARL) framework designed to intelligently manage charging and discharging cycles of Vehicle-to-Grid (V2G)-enabled electric vehicles within energy communities that utilize a shared energy storage system (ESS). By dynamically adapting to fluctuating energy demands, renewable generation, and electricity prices, CoMAD ensures optimal energy efficiency, reduces grid dependency, and minimizes costs without compromising user comfort. The solution has been validated through realistic simulations using real-world household consumption patterns, solar generation data, electric vehicle behavior, and electricity pricing in Slovenia, demonstrating significant improvements in energy autonomy and cost reductions compared to traditional and other intelligent methods.

🚗⚡️

A quick introduction of the minimal setup you need to get a hello world up & running. Required when the project is intended to be distributed as a software package.

## ⚙️ How to Run the Simulation

To execute the simulation, navigate to the project directory and run the following command in your terminal:

```bash
python main.py
```

### 🔸 Select Mode

Choose one of the following operational modes:

- **`immediate`**: Vehicles charge immediately upon arrival.
- **`night`**: Charging restricted to nighttime hours (23:00–08:00); discharging allowed during the day.
- **`drl_single`**: Single-agent Deep Reinforcement Learning (DRL); each household optimizes individually.
- **`comad_v2g`**: Multi-agent DRL (CoMAD); households collaboratively optimize energy usage (recommended).

### 🔸 Number of Households

Specify the number of households to simulate (between **1 and 20**):

```text
Enter number of households (1-20):
```

### 🔸 Community Battery Capacity
Set the shared battery capacity (kWh). Allowed values:

```text
50, 100, 150, 200, 250, 300, 350
```

###🔸 Training Mode
Choose whether to run in training mode (True) or validation/testing mode (False):

```text
Training? (True/False):
```

## 📂 Code Description

- **`main.py`**: The main entry point for the simulation. It interactively prompts users for configuration parameters, including:
  - **Mode selection** (`immediate`, `night`, `drl_single`, `comad_v2g`)
  - **Number of households** (1–20)
  - **Community battery capacity** (50, 100, 150, 200, 250, 300, 350 kWh)
  - **Training mode** (`True` for training, `False` for validation/testing)

All user inputs are validated during runtime, ensuring the simulation initializes correctly.

**Example usage:**

```bash
Select mode (immediate, night, drl_single, comad_v2g): comad_v2g
Enter number of households (1-20): 10
Enter community battery capacity in kWh [50, 100, 150, 200, 250, 300, 350]: 200
Training? (True/False): False
```

## 📂 Data Sources and Modeling

The simulation utilizes realistic data and proven modeling tools for accurate and replicable results. Specifically, the following datasets and tools were employed:

- **Household Consumption**: Modeled using **LoadProfileGenerator**, an agent-based behavior simulation for realistic household energy consumption patterns [^1].
- **Solar Energy Production**: Solar energy generation was calculated based on direct radiation data from [Open-Meteo](https://open-meteo.com/) for accurate solar production estimates.
- **Electric Vehicle (EV) and V2G**: EV charging behaviors, battery capacities, and energy consumption were generated using **emobpy**, a validated open-source tool designed for realistic battery-electric vehicle simulations [^2].
- **Electricity Prices**: Real electricity market pricing data was sourced from the **BSP SouthPool Energy Exchange**, reflecting realistic Slovenian market conditions [^3].

### 📚 References

[^1]: Pflugradt, N., Stenzel, P., Kotzur, L. & Stolten, D. *LoadProfileGenerator: An agent-based behavior simulation for generating realistic load profiles.* [LoadProfileGenerator Documentation](https://github.com/LoadProfileGenerator/LoadProfileGenerator), 2022.

[^2]: Gaete, C., Kramer, H., Schill, W.-P., & Zerrahn, A. (2021). emobpy: An open tool for creating battery-electric vehicle time series from empirical data. *Scientific Data*, 8(1), 152.

[^3]: BSP SouthPool Energy Exchange. [Day-ahead electricity prices data](https://www.bsp-southpool.com/day-ahead-market.html).

Additionally, solar radiation data was sourced from [Open-Meteo](https://open-meteo.com/).

## 📦 Requirements

Ensure the following Python libraries are installed to successfully run the simulation:

```bash
pip install numpy icecream torch==2.2.2
pip install icecream==2.1.3 torch==2.2.2
```

## Licensing

> The code in this project is licensed under MIT license.


## Acknowledgement

> This work was supported by the HORIZON-MSCA-IF project TimeSmart (No. 101063721) and by the Slovenian Research Agency under grant P2-0016.
Additionally, this work was partially funded by the European Union’s Horizon Europe research and innovation programme under the DEDALUS project (No. 101103998) and the STREAM project (No. 101075654).
