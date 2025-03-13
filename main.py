"""
MIT License

Copyright (c) 2025 Jernej Hribar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import time
import simpy
import numpy as np
import pandas as pd
from transmissionline import TransmissionLine
from household import Household
from sharedbattery import SharedBattery

RANDOM_SEED = 42  # Random seed
NUM_HOUSEHOLDS = 20  # The total number of a household, 20 is maximum
SIM_STEPS = 4 * 24 * 31 + 1 # The amount of time simulation lasts
SIM_STEPS = 4 * 24 * 365 + 1 # The amount of time simulation lasts
TIME_STEP_SIZE = 0.25  # Time step size in an hour, 0.25 = 15 minutes
TRAIN = False  #True  # False we use test(validation) datasets, if True, we are training the system

# NAMES OF THE MODE OF OPERATION
MODE_IMMEDIATE = "immediate"  # charge the vehicle as soon as you arrive at home
MODE_NIGHT = "night"  # Charging is limited to nighttime between 23:00 and 8:00, discharge allowed during day
MODE_DRL_SINGLE = "drl_single"  # Policy based on DRL, single agent, each house optimises by itself
MODE_COMAD = "comad_v2g"  # Multi-agent DRL approach, the one we propose for this paper, and is our solution
AVAILABLE_MODES = [MODE_IMMEDIATE, MODE_NIGHT, MODE_DRL_SINGLE, MODE_COMAD]

# Household and V2G hyperparameters
V2G_CHARGE_EFFICIENCY = .9
V2G_DISCHARGE_EFFICIENCY = .95
V2G_CHARGE_POWER = 7  # The maximal charge or discharge efficiency in KWhs
INSTALLED_PV_SIZE = 30   # The square meters of a household
PV_EFFICIENCY = 0.275  # Overall efficiency of conversion


# Function to get a valid selection mode
def get_sel_mode():
    while True:
        mode = input(f"Select mode ({', '.join(AVAILABLE_MODES)}): ").strip()
        if mode in AVAILABLE_MODES:
            return mode
        else:
            print("Invalid mode. Please choose from:", AVAILABLE_MODES)


# Function to get a valid number of households
def get_num_households():
    while True:
        try:
            num = int(input("Enter number of households (1-20): "))
            if 1 <= num <= 20:
                return num
            else:
                print("Number must be between 1 and 20.")
        except ValueError:
            print("Please enter a valid integer.")


# Function to get a valid community battery capacity
def get_com_bat_cap():
    while True:
        try:
            cap = int(input("Enter community battery capacity in kWh [50, 100, 150, 200, 250, 300, 350): "))
            if 50 <= cap <= 350:
                return cap
            else:
                print("Capacity must be an integer between 50 and 350.")
        except ValueError:
            print("Please enter a valid integer.")


# Function to get a valid training flag
def get_training_flag():
    while True:
        train_input = input("Training? (True/False): ").strip().lower()
        if train_input in ['true', 'false']:
            return train_input == 'true'
        else:
            print("Please enter 'True' or 'False'.")


if __name__ == '__main__':
    time_start = time.time()
    print("The Simulation Started.")
    np.random.seed(RANDOM_SEED)  #

    # Prompt user input with validation
    sel_mode = get_sel_mode()
    num_hh = get_num_households()
    com_bat_cap = get_com_bat_cap()
    training = get_training_flag()
    # Initialize the simulation
    env = simpy.Environment()  # Create simulation en

    # Create a shared resource, Creating the community power line to share energy
    community_power_line = TransmissionLine(env, num_households=num_hh)

    # Init Shared battery
    shared_battery = SharedBattery(unique_id=0, battery_cap=com_bat_cap, timestep_size=TIME_STEP_SIZE,
                                   num_households=num_hh, train=training)

    # Init Households
    households = []
    for i in range(num_hh):
        households.append(Household(unique_id=i, timestep_size=TIME_STEP_SIZE, mode=sel_mode,
                                    v2g_charge_efficiency=V2G_CHARGE_EFFICIENCY,
                                    v2g_discharge_efficiency=V2G_DISCHARGE_EFFICIENCY,
                                    v2g_charge_power=V2G_CHARGE_POWER,  pv_size=INSTALLED_PV_SIZE, pv_eff=PV_EFFICIENCY,
                                    train=training, shared_bat=shared_battery
                                    )
                          )

    # Append processes to environment for households:
    for household in households:
        env.process(household.v2g_charging(env))
        env.process(household.energy_household_collects_and_consumes(env))
        env.process(household.v2g_update_values(env))
        env.process(household.energy_exchange(env, community_power_line))
        env.process(household.collect_simulation_data(env))
        env.process(household. electricity_cost(env))
        # Adding DRL related Processes to the file
        env.process(household.drl_train(env))
        env.process(household.track_el_cost(env))

    # Append processes to environment for shared batter
    env.process(shared_battery.charge_or_discharge(env, community_power_line))
    env.process(shared_battery.local_grid_energy_exchange(env))
    env.process(shared_battery.energy_in_battery(env))
    env.process(shared_battery.collect_simulation_data(env))
    env.process(shared_battery.dynamic_energy_price_change(env))

    # Run simulation
    env.run(SIM_STEPS)

    # Collect data, after the simulation is over for every household
    com_hh_energy_cons = []
    com_hh_energy_col = []
    com_hh_v2g_energy_cons = []
    com_hh_v2g_downtime = []
    com_hh_v2g_energy_deficit = []
    com_hh_cost = []
    com_hh_sell = []
    com_hh_buy = []

    if not training:
        for household in households:
            hh_energy_cons = household.monthly_energy_cons  # Energy consumption of the household
            hh_energy_col = household.monthly_energy_col  # Energy collected by the household
            hh_v2g_energy_cons = household.monthly_energy_cons_v2g1  # Energy consumption
            hh_v2g_downtime = household.monthly_ts_no_energy_v2g1 # Downtime, no energy available in the electric car
            hh_v2g_energy_deficit = household.month_v2g_energy_deficit  # How much energy is missing in the vehicle
            hh_cost = household.monthly_hh_cost  # cost of the household as it would be an individual on the market
            hh_sell = household.monthly_hh_energy_sell  # selling el. as it would following individual pricing
            hh_buy = household.monthly_hh_energy_buy  # buying el. as it would following individual pricing

            # Save values to the dataframe and write to .csv file
            df_hh = pd.DataFrame()
            df_hh['en_cons'] = np.round(hh_energy_cons, 2)
            df_hh['en_col'] = np.round(hh_energy_col, 2)
            df_hh['v2g_en_cons'] = np.round(hh_v2g_energy_cons, 2)
            df_hh['v2g_en_down'] = np.round(hh_v2g_downtime, 2)
            df_hh['v2g_en_def'] = np.round(hh_v2g_energy_deficit, 2)
            df_hh['cost'] = np.round(hh_cost, 2)
            df_hh['sell'] = np.round(hh_sell, 2)
            df_hh['buy'] = np.round(hh_buy, 2)
            df_hh.to_csv('results/' + sel_mode + '/household_' + str(household.unique_id) + 'info_for_num_households_'
                         + str(num_hh) + '_bat_size_' + str(com_bat_cap) + '.csv')

            # Append values to the combined list
            com_hh_energy_cons.append(hh_energy_cons)
            com_hh_energy_col.append(hh_energy_col)
            com_hh_v2g_energy_cons.append(hh_v2g_energy_cons)
            com_hh_v2g_downtime.append(hh_v2g_downtime)
            com_hh_v2g_energy_deficit.append(hh_v2g_energy_deficit)
            com_hh_cost.append(hh_cost)
            com_hh_sell.append(hh_sell)
            com_hh_buy.append(hh_buy)

        # save combined households information
        df_hh = pd.DataFrame()
        df_hh['en_cons'] = np.round([sum(i) for i in zip(*com_hh_energy_cons)], 2)
        df_hh['en_col'] = np.round([sum(i) for i in zip(*com_hh_energy_col)], 2)
        df_hh['v2g_en_cons'] = np.round([np.average(i) for i in zip(*com_hh_v2g_energy_cons)], 2)
        df_hh['v2g_en_down'] = np.round([np.average(i)for i in zip(*com_hh_v2g_downtime)], 2)
        df_hh['v2g_en_def'] = np.round([np.average(i) for i in zip(*com_hh_v2g_energy_deficit)], 2)
        df_hh['cost'] = np.round([sum(i) for i in zip(*com_hh_cost)], 2)
        df_hh['sell'] = np.round([sum(i) for i in zip(*com_hh_sell)], 2)
        df_hh['buy'] = np.round([sum(i) for i in zip(*com_hh_buy)], 2)
        df_hh.to_csv('results/' + sel_mode + '/households_info_for_num_households_'
                     + str(num_hh) + '_bat_size_' + str(com_bat_cap) + '.csv')

        # Save in a quarters, combine 3 months as one
        df_hh = pd.DataFrame()
        df_hh['en_cons'] = np.round([sum(x) for x in [[sum(i) for i in zip(*com_hh_energy_cons)][i * 3:(i + 1) * 3] for i in range(4)]], 2)
        df_hh['en_col'] = np.round([sum(x) for x in [[sum(i) for i in zip(*com_hh_energy_col)][i * 3:(i + 1) * 3] for i in range(4)]], 2)
        df_hh['v2g_en_cons'] = np.round([np.average(x) for x in [[np.average(i) for i in zip(*com_hh_v2g_energy_cons)][i * 3:(i + 1) * 3] for i in range(4)]], 2)
        df_hh['v2g_en_down'] = np.round([np.average(x) for x in [[np.average(i) for i in zip(*com_hh_v2g_downtime)][i * 3:(i + 1) * 3] for i in range(4)]], 2)
        df_hh['v2g_en_def'] = np.round([np.average(x) for x in [[np.average(i) for i in zip(*com_hh_v2g_energy_deficit)][i * 3:(i + 1) * 3] for i in range(4)]], 2)
        df_hh['cost'] = np.round([sum(x) for x in [[sum(i) for i in zip(*com_hh_cost)][i * 3:(i + 1) * 3] for i in range(4)]], 2)
        df_hh['sell'] = np.round([sum(x) for x in [[sum(i) for i in zip(*com_hh_sell)][i * 3:(i + 1) * 3] for i in range(4)]], 2)
        df_hh['buy'] = np.round([sum(x) for x in [[sum(i) for i in zip(*com_hh_buy)][i * 3:(i + 1) * 3] for i in range(4)]], 2)
        df_hh.to_csv('results/' + sel_mode + '/quarterly_households_info_for_num_households_'
                     + str(num_hh) + '_bat_size_' + str(com_bat_cap) + '.csv')

        # Collect DATA from Shared battery, create a pandas dataframe
        sh_energy_ex_int = shared_battery.monthly_energy_difference  # amount of energy exchanged inside the community
        sh_energy_trans_to_grid = shared_battery.monthly_external_grid_diff_pos  # amount of energy sold to the grid
        sh_energy_trans_from_grid = shared_battery.monthly_external_grid_diff_neg  # amount of energy bought) from the grid
        sh_energy_grid_diff = [x + y for x, y in zip(sh_energy_trans_to_grid, sh_energy_trans_from_grid)]
        sh_sell_energy = shared_battery.monthly_energy_sell  # Money we gain from selling to the grid
        sh_buy_energy = shared_battery.monthly_energy_buy  # Cost of buying electricity
        sh_combined = [x + y for x, y in zip(sh_sell_energy, sh_buy_energy)]
        sh_life_cycles = shared_battery.bat_life_cycles_list  # charging cycles of the shared battery

        # Save to .csv file for the shared battery, each month separately
        df_bat = pd.DataFrame()
        df_bat['sh_en_ex_int'] = np.round(sh_energy_ex_int, 2)
        df_bat['sh_en_trans_to_grid'] = np.round(sh_energy_trans_to_grid, 2)
        df_bat['sh_en_trans_from_grid'] = np.round(sh_energy_trans_from_grid, 2)
        df_bat['sh_en_grid_diff'] = np.round(sh_energy_grid_diff, 2)
        df_bat['sh_sell_en'] = np.round(sh_sell_energy, 2)
        df_bat['sh_buy_en'] = np.round(sh_buy_energy, 2)
        df_bat['sh_sum_cost'] = np.round(sh_combined, 2)
        df_bat['sh_life_cycles'] = np.round(sh_life_cycles, 2)
        df_bat.to_csv('results/' + sel_mode + '/sh_bat_info_for_num_households_' + str(num_hh) +
                      '_bat_size_' + str(com_bat_cap) + '.csv')

        # Save in a quarters, combine 3 months as one
        df_bat = pd.DataFrame()
        df_bat['sh_en_ex_int'] = np.round([sum(x) for x in [sh_energy_ex_int[i*3:(i+1)*3] for i in range(4)]], 2)
        df_bat['sh_en_trans_to_grid'] = np.round([sum(x) for x in [sh_energy_trans_to_grid[i*3:(i+1)*3] for i in range(4)]], 2)
        df_bat['sh_en_trans_from_grid'] = np.round([sum(x) for x in [sh_energy_trans_from_grid[i*3:(i+1)*3] for i in range(4)]], 2)
        df_bat['sh_en_grid_diff'] = np.round([sum(x) for x in [sh_energy_grid_diff[i*3:(i+1)*3] for i in range(4)]], 2)
        df_bat['sh_sell_en'] = np.round([sum(x) for x in [sh_sell_energy[i*3:(i+1)*3] for i in range(4)]], 2)
        df_bat['sh_buy_en'] = np.round([sum(x) for x in [sh_buy_energy[i*3:(i+1)*3] for i in range(4)]], 2)
        df_bat['sh_sum_cost'] = np.round([sum(x) for x in [sh_combined[i*3:(i+1)*3] for i in range(4)]], 2)
        df_bat['sh_life_cycles'] = np.round([x[2] for x in [sh_life_cycles[i*3:(i+1)*3] for i in range(4)]], 2)
        df_bat.to_csv('results/' + sel_mode + '/quarterly_sh_bat_info_for_num_households_' + str(num_hh) +
                      '_bat_size_' + str(com_bat_cap) + '.csv')

    time_end = time.time()
    time_total = time_end - time_start
    print("Simulation Ended and lasted for ", time_total)
