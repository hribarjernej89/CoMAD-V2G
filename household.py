import numpy as np
from icecream import ic
from dqn import DQNAgent
from collections import deque

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # This is a non-leap year( data is 2022 and 2023)
ELECTRIC_VEHICLE_CONSUMPTION = 0.2  # We assume that electric vehicle consumes .2 KWh per km
ELECTRIC_VEHICLE_FULL_THRESHOLD = .95  # the max level we still decide to charge

# NAMES OF THE MODE OF OPERATION
MODE_IMMEDIATE = "immediate"  # charge the vehicle as soon as you arrive at home
MODE_ALTER = "alternating"  # alternate between available actions, i.e., random policy
MODE_NIGHT = "night"  # Charging is limited to nigh-time between 23:00 and 8:00, discharge allowed during day, such a way we can take advantage of lower prices during night
MODE_DRL_SINGLE = "drl_single"  # Policy based on DRL, single agent, each house optimises by itself
MODE_COMAD = "comad_v2g"  # Proposed solution

# night-mode thresholds,
NIGHT_MODE_DISCHARGE_LEVEL = .5  # The minimum percentage of battery we need to allow discarge during day
NIGHT_MODE_HOURS = [23.0, 24.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # The list of hours during which we allow charge

# AVAILABLE ACTIONS, five actions, each agent and solution has at its disposa,
CHARGE_V2G = 0
HALF_CHARGE_V2G = 1
DISCHARGE_V2G = 2
HALF_DISCHARGE_V2G = 3
NO_CHANGE_V2G = 4

NUM_ACTIONS = 5  # Total number of actions

# DRL REWARD RELATED HYPERPARAMETERS
DRL_MIN_BAT = 0.33  # The minimal battery level we allow in the V2G without penalty

# Deep RL related HYPER-PARAMETERS
DRL_WINDOW_SIZE = 24 * 4  # The window size we will use for input state information
DRL_TRAIN_TIME = 1  # 4  # We train the agent every N steps if necessary
STATE_CONS_NORM = 1.5  # We clip consumption over one 1kh per time step and normalise over it
STATE_PV_NORM = 2.0  # We clip PV generation over one 1.5kh per time step and normalise over it
STATE_ENERGY_EX_NORM = 2.0  # We clip energy exchange between community and grid
STATE_COST_MAX_NORM = .10  # Normalise for 10 cents per kWh when buying and selling
STATE_COST_MIN_NORM = -.025  # Normalise buy prices, note that
REWARD_CONS_RANGE = 1.0  # We clip the consumption reward
DRL_RWD_COST_MAX_NORM = .2  # 20 cents earned in a time step
DRL_RWD_COST_MIN_NORM = -.5  # 50 cents spend in a tem step
ic.configureOutput(includeContext=True)
DRL_RWD_MIN_BAT = 0.25  # This is battery related to reward of the agent
DRL_RWD_MAX_BAT = .8  # The maximal battery level we allow in the V2G without penalty


class Household:    
    def __init__(self, unique_id=0, timestep_size=0.25, mode=MODE_IMMEDIATE,
                 v2g_charge_efficiency=0.9, v2g_discharge_efficiency=0.95,
                 v2g_charge_power=7,  pv_size=1, pv_eff=0.8, train=True, shared_bat=np.nan):
        self.unique_id = unique_id  # Each household has an unique identifier, that matches id the consumption file

        # Simulation parameters
        self.timestep_size = timestep_size  # time step size in hours

        # selected mode of charging for V2G vehicle
        self.mode = mode

        # Shared battery object, to get the necessary information from it with ease
        self.shared_battery = shared_bat

        # Energy parameters
        self.energy_difference_household = 0  # Energy difference between consumption and PV generation
        self.community_energy_transfer = 0  # The amount of energy the household transfer from the comm

        # Read files with simulated values of consumption and power generation
        self.train = train  # If True, the system trains the agents
        # Reading real value from a file, save it to the
        if self.train:
            self.household_consumption_file = open('household_loads/household' + str(self.unique_id) + '_train.csv', 'r')
            self.pv_generation_file = open('PV/PV_data_slo.csv', 'r')
            self.pv_gen_line = 1
            self.v2g1_consumption_file = open('el_veh/household' + str(self.unique_id) + '_large_veh_train.csv', 'r')
            #self.v2g2_consumption_file = open('el_veh/household' + str(self.unique_id) + '_small_veh_train.csv', 'r')
            self.el_cost_file = open('energy_price/comed_price_hour_mean_2022.csv', 'r')

        else:
            self.household_consumption_file = open('household_loads/household' + str(self.unique_id) + '_test.csv', 'r')
            self.pv_generation_file = open('PV/PV_data_slo.csv', 'r')
            # TO DO: set a read line to second year
            self.pv_gen_line = 365 * 24 + 1  # To get to the right line
            self.v2g1_consumption_file = open('el_veh/household' + str(self.unique_id) + '_large_veh_test.csv', 'r')
           #self.v2g2_consumption_file = open('el_veh/household' + str(self.unique_id) + '_small_veh_test.csv', 'r')
            self.el_cost_file = open('energy_price/comed_price_hour_mean_2023.csv', 'r')

        # Read files
        self.household_consumption = self.household_consumption_file.readlines()
        self.pv_generation = self.pv_generation_file.readlines()
        self.v2g1_raw_data = self.v2g1_consumption_file.readlines()
        #self.v2g2_raw_data = self.v2g2_consumption_file.readlines()
        self.el_cost_data = self.el_cost_file.readlines()
        # set lines for reading
        self.household_cons_line = 1
        self.v2g_cons_line = 1
        self.el_cost_line = 1
        # read real values from files
        self.household_consumption_values = self.household_consumption[self.household_cons_line]
        self.pv_generation_values = self.pv_generation[self.pv_gen_line]
        self.v2g1_consumption_values = self.v2g1_raw_data[self.v2g_cons_line]
        #self.v2g2_consumption_values = self.v2g2_raw_data[self.v2g_cons_line]
        # Obtain the values we need for consumption and
        self.hh_cons = np.round(float(self.household_consumption_values.split(',')[2]), 5)  # Household consumption
        self.pv_gen_power = np.round(float(self.pv_generation_values.split(',')[2]), 2)
        #print("Line:", self.pv_generation_values, "value:", self.pv_gen_power)
        self.pv_size = pv_size  # Installed peak power in square meters KW, we multiply by this constant
        self.pv_eff = pv_eff  # Efficiency of power conversion
        self.pv_gen_energy = np.round(self.pv_gen_power * self.pv_eff * self.pv_size * self.timestep_size * 0.001, 5)
        # get the cost of energy

        self.el_cost_values = self.el_cost_data[self.el_cost_line]
        self.el_buy_cost = np.round(float(self.el_cost_values.split(',')[4]), 5)
        self.el_sell_cost = np.round(float(self.el_cost_values.split(',')[3]), 5)
        self.hh_energy_cost = 0 # to track household cost based on the energy prices
        #print("Values of el. cost:",self.el_cost_values, "Buy:", self.el_buy_cost, "sell:", self.el_sell_cost)

        #if v2g_num in [0, 1, 2]:
        #    self.v2g_num = v2g_num  # The number of electric vehicles the household has
        #else:
        #    print("Selected number of vehicles is not zero, one, or two! Number of Electric vehicles is set to two!")
        #    self.v2g_num = 2

        # V2G parameters for electric vehicle 1
        self.v2g1_capacity = float(self.v2g1_consumption_values.split(',')[5])  # The capacity of the vehicle battery in kWs as was original car battery
        self.v2g1_available_energy = self.v2g1_capacity  # Init values are full vehicle batteries
        self.v2g1_percentage_full = np.round(self.v2g1_available_energy / self.v2g1_capacity, 3)  # energy in percentage
        #self.v2g1_energy_difference = 0  # Energy difference in v2g element
        self.v2g1_read_value1 = self.v2g1_consumption_values.split(',')[1]
        self.v2g1_ts_no_energy = 0  # counting the number of steps v2g has no energy available
        #self.v2g1_energy_diff = .0
        if self.v2g1_read_value1 == 'True':
            self.v2g1_is_present = True
            self.v2g1_is_present_past = True
        else:
            self.v2g1_is_present = False
            self.v2g1_is_present_past = False
        self.v2g1_left = False
        #self.v2g1_consumption = np.round(float(self.v2g1_consumption_values.split(',')[3]), 5)
        self.v2g1_consumption = np.round(float(self.v2g1_consumption_values.split(',')[2]),
                                         6) * ELECTRIC_VEHICLE_CONSUMPTION
        self.v2g1_read_value2 = self.v2g1_consumption_values.split(',')[7]
        if self.v2g1_read_value2 == 'True':
            self.v2g1_is_back = True
        else:
            self.v2g1_is_back = False

        # tracking energy deficit of V2G, i.e., energy the vehicle does not have but we need
        self.v2g_energy_deficit = 0.0

        # Efficiency of energy charging and discharging
        self.v2g_charge_efficiency = v2g_charge_efficiency  # The efficiency of charging
        self.v2g_discharge_efficiency = v2g_discharge_efficiency  # The efficiency of discharging

        # Parameters we need for the V2G
        self.v2g_energy_difference = 0
        self.v2g_charge_power = v2g_charge_power  # This value is kW
        self.v2g_charge_energy = self.v2g_charge_power * self.timestep_size  # in kWh
        print("Energy required to charge battery in a time-step", self.v2g_charge_energy)

        # parameters to measure performance, we start on January first at 00:01, if not change accordingly
        self.current_month = 0
        self.day_in_a_month = 0
        self.day_time_step_len = int(24 / self.timestep_size)
        self.hour_time_step = 0
        self.hour_in_day = 0

        # DRL agent related things, TO DO: copy from past works, how to do it best? check my past work
        if self.mode == MODE_DRL_SINGLE or self.mode == MODE_COMAD:
            self.drl_is_used = True
            print("We are using DRL")
            # State space will consist of one day of data for each variable

            self.drl_state_household_cons = deque(np.zeros(DRL_WINDOW_SIZE), maxlen=DRL_WINDOW_SIZE)  # H. consumption
            self.drl_state_pv_gen_energy = deque(np.zeros(DRL_WINDOW_SIZE), maxlen=DRL_WINDOW_SIZE)  # Energy we gen.
            self.drl_state_v2g1_charge = deque(np.zeros(DRL_WINDOW_SIZE), maxlen=DRL_WINDOW_SIZE)  # Charging profile
            self.drl_state_v2g1_energy = deque(np.zeros(DRL_WINDOW_SIZE), maxlen=DRL_WINDOW_SIZE)  # Energy in V2G
            self.drl_state_shared_bat = deque(np.zeros(DRL_WINDOW_SIZE), maxlen=DRL_WINDOW_SIZE)  # Shared bat. energy
            self.drl_state_shared_en_ex = deque(np.zeros(DRL_WINDOW_SIZE), maxlen=DRL_WINDOW_SIZE)  # Energy exchanged
            self.drl_state_sell_price = deque(np.zeros(DRL_WINDOW_SIZE), maxlen=DRL_WINDOW_SIZE)  # Sell cost of energy
            self.drl_state_buy_price = deque(np.zeros(DRL_WINDOW_SIZE), maxlen=DRL_WINDOW_SIZE)  # Buy cost of energy

            # Tracking max number of consumption, TO DO: REMOVE tracking_* FROM Simulation
            self.tracking_household_consumption = []
            self.tracking_household_pv_gen = []
            self.tracking_shared_bat_energy_exchange = []

            if self.mode == MODE_DRL_SINGLE:
                print("We are initialising Single Agent approach!")
                self.drl_agent_v2g1_state = np.array([self.drl_state_household_cons, self.drl_state_pv_gen_energy,
                                                      self.drl_state_v2g1_charge, self.drl_state_v2g1_energy,
                                                      self.drl_state_sell_price, self.drl_state_buy_price
                                                      ]).flatten()
            else: # We are using multi-agent approach
                print("We are initialising MULTI-AGENT approach!")
                self.drl_agent_v2g1_state = np.array([self.drl_state_household_cons, self.drl_state_pv_gen_energy,
                                                      self.drl_state_v2g1_charge, self.drl_state_v2g1_energy,
                                                      self.drl_state_shared_bat, self.drl_state_shared_en_ex,
                                                      self.drl_state_sell_price, self.drl_state_buy_price
                                                      ]).flatten()

            print("DRL state init:", self.drl_state_household_cons)
            # Init with empty states
            self.drl_agent_v2g1_past_state = self.drl_agent_v2g1_state

            self.drl_agent_v2g1_action = CHARGE_V2G
            self.drl_agent_v2g1_reward = 0

            # Init the DRL Agents, one for each V2G
            print("Testing State space size:", 5* DRL_WINDOW_SIZE, "vs.", len(self.drl_agent_v2g1_state))
            self.drl_agent_v2g1 = DQNAgent(n_states=len(self.drl_agent_v2g1_state), n_actions=NUM_ACTIONS)
            # Try to load pre/trained Network if available.
            try:
                self.drl_agent_v2g1.load("drl_trained_nets/" + self.mode + "_total_num_" +
                                         str(self.shared_battery.num_households) + "_household_" + str(self.unique_id)
                                         + "_bat_size_" + str(self.shared_battery.battery_cap) + ".pkl")
                #self.drl_agent_v2g1.save("drl_trained_nets/" + self.mode + "household_" + str(self.unique_id) + ".pkl")
                #print("Loaded trained net:", self.mode + "_household_" + str(self.unique_id) + ".pkl")
                self.drl_agent_v2g1.epsilon = 0.2
            except FileNotFoundError:
                print("No trained agent available!")
            except:
                print("Something went wrong with loading trained agent!")


        else:
            self.drl_is_used = False
            # We need the following just to ensure everything is correct with calling v2g action function
            self.drl_agent_v2g1 = np.nan
            self.drl_agent_v2g1_state = 0
            # TO DO: Do for the other agent

        # list of information we are collecting during simulation
        self.track_rwd = [0]  # Init with one value
        self.track_energy_price_battery = []
        self.track_energy_price_household = []
        self.track_household_energy_price_buy = []
        self.track_household_energy_price_sell = []
        self.track_shared_bat_price_sell = []
        self.track_sharedd_bat_price_buy = []
        # Energy consumption
        self.temp_energy_cons = []
        self.daily_energy_cons = []
        self.monthly_energy_cons = []
        # Energy Collected
        self.temp_energy_col = []
        self.daily_energy_col = []
        self.monthly_energy_col = []
        # Energy Consumption V2G vehicle 1
        self.temp_energy_cons_v2g1 = []
        self.daily_energy_cons_v2g1 = []
        self.monthly_energy_cons_v2g1 = []
        self.monthly_ts_no_energy_v2g1 = []
        # Energy charged V2G vehicle 1
        self.temp_energy_charge_v2g1 = []
        self.daily_energy_charge_v2g1 = []
        self.monthly_energy_charge_v2g1 = []
        # Tracking Energy Deficit of V2G
        self.temp_v2g_energy_deficit = []
        self.daily_v2g_energy_deficit = []
        self.month_v2g_energy_deficit = []
        # Individual cost of electricity,
        self.temp_hh_cost = []
        self.daily_hh_cost = []
        self.monthly_hh_cost = []
        self.temp_hh_energy_sell = []
        self.daily_hh_energy_sell = []
        self.monthly_hh_energy_sell = []
        self.temp_hh_energy_buy = []
        self.daily_hh_energy_buy = []
        self.monthly_hh_energy_buy = []

        self.test_list = []

    def energy_household_collects_and_consumes(self, env):
        """
        Process for collecting energy from the photovoltaics system and energy
        """
        while True:
            yield env.timeout(1)   # In each time step we will collect and consume energy
            # Make sure we mark the household consumption
            self.household_consumption_values = self.household_consumption[self.household_cons_line]
            self.hh_cons = np.round(float(self.household_consumption_values.split(',')[2]), 5)
            self.test_list.append(self.hh_cons)
            # print("Household consumption:", self.hh_cons, np.max(self.test_list))

            # Now we determine the amount of energy the PV generate in a step
            self.pv_generation_values = self.pv_generation[self.pv_gen_line]
            self.pv_gen_power = np.round(float(self.pv_generation_values.split(',')[2]), 6)
            self.pv_gen_energy = np.round(self.pv_gen_power * self.pv_size * self.pv_eff * self.timestep_size * 0.001, 5)
            self.energy_difference_household = self.pv_gen_energy - self.hh_cons
            # Update new lines for next step
            self.household_cons_line += 1
            if env.now % int(1/self.timestep_size) == 0:  # PV generation data line
                self.pv_gen_line += 1
            #print("Energy difference is:", self.energy_difference_household)
            # We also need to add to collect
            self.temp_energy_col.append(self.pv_gen_energy)
            self.temp_energy_cons.append(self.hh_cons)

            self.test_list.append(-self.pv_gen_energy - self.hh_cons)

    def v2g_update_values(self, env):
        """
        Process reading values of consumption and setting the state for both vehicles controlling and managing vehicle-to-grid elements. Each household can have up two electric vehicles
        present in the system.
        """
        while True:
            yield env.timeout(1)  # In each time step we will decide what to do with the energy
            #print("We are at time-step", env.now, " Household", self.unique_id, "Process Energy V2G element")
            # to track hour in day
            self.hour_time_step += self.timestep_size
            self.hour_in_day = np.floor(self.hour_time_step)
            #print("hour is:", self.hour_in_day)
            # First we read from the file to update the values for both vehicles
            # print("line to read", self.v2g_cons_line)
            self.v2g1_consumption_values = self.v2g1_raw_data[self.v2g_cons_line]
            #self.v2g2_consumption_values = self.v2g2_raw_data[self.v2g_cons_line]
            self.v2g_cons_line = self.v2g_cons_line + 1

            # Here we update presence from the previous time steo
            self.v2g1_is_present_past = self.v2g1_is_present
            #self.v2g2_is_present_past = self.v2g2_is_present

            # Update V2G parameters for electric vehicle 1
            self.v2g1_read_value1 = self.v2g1_consumption_values.split(',')[1]
            if self.v2g1_read_value1 == 'True':
                self.v2g1_is_present = True
            else:
                self.v2g1_is_present = False
            #self.v2g1_consumption = np.round(float(self.v2g1_consumption_values.split(',')[3]), 6) * 40
            self.v2g1_consumption = np.round(float(self.v2g1_consumption_values.split(',')[2]),
                                             6) * ELECTRIC_VEHICLE_CONSUMPTION

            self.v2g1_read_value2 = self.v2g1_consumption_values.split(',')[7]
            if self.v2g1_read_value2 == 'True':
                self.v2g1_is_back = True
            else:
                self.v2g1_is_back = False

            if not self.v2g1_is_present and self.v2g1_is_present_past:
                self.v2g1_left = True
                #print("Vehicle 1 left the house", self.v2g1_is_present, self.v2g1_is_present_past, "at", env.now)
            else:
                self.v2g1_left = False

            if self.v2g1_available_energy - self.v2g1_consumption >= 0.0:
                self.v2g1_available_energy = self.v2g1_available_energy - self.v2g1_consumption
                self.v2g_energy_deficit = 0.0
            else:
                self.v2g_energy_deficit = self.v2g1_available_energy - self.v2g1_consumption
                self.v2g1_available_energy = 0.0
            self.v2g1_percentage_full = np.round(self.v2g1_available_energy / self.v2g1_capacity, 3)

            if self.v2g1_available_energy == 0.0:
                #print("Vehicle 1 is without energy!!!")
                self.v2g1_ts_no_energy += 1

            # Save the consumption of v2g Vehicles for simulation tracking into a list...
            self.temp_energy_cons_v2g1.append(self.v2g1_consumption)
            self.temp_v2g_energy_deficit.append(self.v2g_energy_deficit)

    def v2g_charging(self, env):
        """
        Process that controls charging and discharging of electric vehicles. TO DO: Make it more modular...
        """
        while True:
            yield env.timeout(1)  # In each time step we may need to exchange the energy with
            if self.drl_is_used:
                # Save information for the update state space from agent:
                self.drl_state_household_cons.append(np.clip(self.hh_cons, 0.0, STATE_CONS_NORM))  # Normalised
                self.drl_state_pv_gen_energy.append(np.clip(self.pv_gen_energy, 0.0, STATE_PV_NORM) / STATE_PV_NORM)  # Normalised
                self.drl_state_v2g1_charge.append(self.v2g1_percentage_full)  # normalised
                self.drl_state_v2g1_energy.append(self.v2g_energy_difference / self.v2g_charge_energy)  # normalised
                # Save information from Shared Battery
                self.drl_state_shared_bat.append(self.shared_battery.battery_energy_percentage)  #
                self.drl_state_shared_en_ex.append(np.clip(self.shared_battery.energy_difference/self.shared_battery.num_households, -STATE_ENERGY_EX_NORM, STATE_ENERGY_EX_NORM) / STATE_ENERGY_EX_NORM)
                if self.mode == MODE_DRL_SINGLE:

                    self.drl_state_buy_price.append(np.round(np.clip(self.el_buy_cost,STATE_COST_MIN_NORM,STATE_COST_MAX_NORM)/(STATE_COST_MAX_NORM - STATE_COST_MIN_NORM), 2))
                    self.drl_state_sell_price.append(np.round(np.clip(self.el_sell_cost,STATE_COST_MIN_NORM,STATE_COST_MAX_NORM)/(STATE_COST_MAX_NORM - STATE_COST_MIN_NORM), 2))
                    #self.drl_state_cost.append(np.clip(self.hh_energy_cost ,STATE_COST_MIN_NORM,STATE_COST_MAX_NORM))
                elif self.mode == MODE_COMAD:
                    self.drl_state_buy_price.append(np.round(
                        np.clip(self.shared_battery.buy_energy_price, STATE_COST_MIN_NORM, STATE_COST_MAX_NORM) / (
                                    STATE_COST_MAX_NORM - STATE_COST_MIN_NORM), 2))
                    self.drl_state_sell_price.append(np.round(
                        np.clip(self.shared_battery.sell_energy_price, STATE_COST_MIN_NORM, STATE_COST_MAX_NORM) / (
                                    STATE_COST_MAX_NORM - STATE_COST_MIN_NORM), 2))
                else:
                    print("Error in cost state:")

            self.v2g_energy_difference = 0
            household_energy_change = self.pv_gen_energy - self.hh_cons
            if self.v2g1_is_present:  # TO DO: check if in other modes charging of V2G goes over 100 percent...
                if self.drl_is_used:

                    if self.mode == MODE_DRL_SINGLE:
                        self.drl_agent_v2g1_state = np.array(
                            [self.drl_state_household_cons, self.drl_state_pv_gen_energy,
                             self.drl_state_v2g1_charge, self.drl_state_v2g1_energy,
                             self.drl_state_sell_price, self.drl_state_buy_price
                             ]).flatten()
                    else:  # We are using multi-agent approach
                        self.drl_agent_v2g1_state = np.array(
                            [self.drl_state_household_cons, self.drl_state_pv_gen_energy,
                             self.drl_state_v2g1_charge, self.drl_state_v2g1_energy,
                             self.drl_state_shared_bat, self.drl_state_shared_en_ex,
                             self.drl_state_sell_price, self.drl_state_buy_price
                             ]).flatten()

                    if not self.v2g1_is_back:
                        #print("We are saving an experience")
                        # Determine the reward:
                        if self.mode == MODE_DRL_SINGLE:
                            household_energy_change = self.pv_gen_energy - self.hh_cons + self.v2g_energy_difference
                            self.drl_agent_v2g1_reward = rwd_drl_single(self.v2g1_percentage_full,
                                                                        household_energy_change)
                        elif self.mode == MODE_COMAD:
                            self.drl_agent_v2g1_reward = rwd_drl(self.v2g1_percentage_full,
                                                                 self.shared_battery.community_exchange_with_grid /
                                                                 self.shared_battery.num_households,
                                                                 self.shared_battery.battery_energy_step_cost /
                                                                 self.shared_battery.num_households
                                                                 )
                        else:
                            print("Unknown DRL mode selected")

                        # Store the experience

                        self.drl_agent_v2g1.remember(self.drl_agent_v2g1_past_state, self.drl_agent_v2g1_action,
                                                     self.drl_agent_v2g1_reward, self.drl_agent_v2g1_state)
                        self.track_rwd.append(self.drl_agent_v2g1_reward)

                v2g1_action = v2g_action(mode=self.mode, percentage_full=self.v2g1_percentage_full, env_time=env.now,
                                         hour_in_day=self.hour_in_day, change_of_energy=household_energy_change,
                                         v2g_charge_energy=self.v2g_charge_energy, drl_agent=self.drl_agent_v2g1,
                                         drl_state=self.drl_agent_v2g1_state)

                match v2g1_action:
                    case 0: #CHARGE_V2G:
                        if (self.v2g1_available_energy + self.v2g_charge_energy * self.v2g_charge_efficiency) <= self.v2g1_capacity:
                            self.v2g1_available_energy += self.v2g_charge_energy * self.v2g_charge_efficiency
                            self.v2g_energy_difference -= self.v2g_charge_energy
                        else:
                            energy_change = self.v2g1_capacity - self.v2g1_available_energy
                            self.v2g1_available_energy += energy_change
                            self.v2g_energy_difference -= energy_change / self.v2g_charge_efficiency
                    case 1: #HALF_CHARGE_V2G:
                        if (self.v2g1_available_energy + self.v2g_charge_energy * self.v2g_charge_efficiency) <= (.5 * self.v2g1_capacity):
                            self.v2g1_available_energy += .5 * self.v2g_charge_energy * self.v2g_charge_efficiency
                            self.v2g_energy_difference -= .5 * self.v2g_charge_energy
                        else:
                            energy_change = self.v2g1_capacity - self.v2g1_available_energy
                            self.v2g1_available_energy += energy_change
                            self.v2g_energy_difference -= energy_change / self.v2g_charge_efficiency
                    case 2: #DISCHARGE_V2G:
                        if self.v2g1_percentage_full > DRL_MIN_BAT: # Limitation of the system
                            self.v2g1_available_energy -= self.v2g_charge_energy * self.v2g_charge_efficiency
                            self.v2g_energy_difference += self.v2g_charge_energy
                    case 3: #HALF_DISCHARGE_V2G:
                        if self.v2g1_percentage_full > DRL_MIN_BAT: # Limitation of the system
                            self.v2g1_available_energy -= .5 * self.v2g_charge_energy * self.v2g_charge_efficiency
                            self.v2g_energy_difference += .5 * self.v2g_charge_energy
                    case 4: #NO_CHANGE_V2G:
                        self.v2g_energy_difference += 0
                self.v2g1_percentage_full = np.round(self.v2g1_available_energy / self.v2g1_capacity, 3)
                if self.drl_is_used:
                    # Save the action we took
                    self.drl_agent_v2g1_action = v2g1_action  # Store selected action for DRL
                    # Save information for the update state space
                    self.drl_agent_v2g1_past_state = self.drl_agent_v2g1_state

            elif self.v2g1_left and self.drl_is_used:
                # We only need to save an experience
                if self.mode == MODE_DRL_SINGLE:
                    self.drl_agent_v2g1_state = np.array(
                        [self.drl_state_household_cons, self.drl_state_pv_gen_energy,
                         self.drl_state_v2g1_charge, self.drl_state_v2g1_energy,
                         self.drl_state_sell_price, self.drl_state_buy_price
                        ]).flatten()
                else:  # We are using multi-agent approach
                    self.drl_agent_v2g1_state = np.array([self.drl_state_household_cons, self.drl_state_pv_gen_energy,
                                                          self.drl_state_v2g1_charge, self.drl_state_v2g1_energy,
                                                          self.drl_state_shared_bat, self.drl_state_shared_en_ex,
                                                          self.drl_state_sell_price, self.drl_state_buy_price
                                                          ]).flatten()

                if self.mode == MODE_DRL_SINGLE:
                    household_energy_change = self.pv_gen_energy - self.hh_cons + self.v2g_energy_difference
                    self.drl_agent_v2g1_reward = rwd_drl(self.v2g1_percentage_full, household_energy_change,
                                                         self.hh_energy_cost)
                elif self.mode == MODE_COMAD:
                    self.drl_agent_v2g1_reward = rwd_drl(self.v2g1_percentage_full,
                                                         self.shared_battery.community_exchange_with_grid /
                                                         self.shared_battery.num_households,
                                                         self.shared_battery.battery_energy_step_cost /
                                                         self.shared_battery.num_households
                                                         )
                else:
                    print("Unknown DRL mode selected")

                # Store the experience
                self.drl_agent_v2g1.remember(self.drl_agent_v2g1_past_state, self.drl_agent_v2g1_action,
                                             self.drl_agent_v2g1_reward, self.drl_agent_v2g1_state)
                self.track_rwd.append(self.drl_agent_v2g1_reward)

    def energy_exchange(self, env, community_power_line):
        """
        Process to exchange the required energy with the shared Community Battery
        """
        while True:
            yield env.timeout(1)  # In each time step we may need to exchange the energy with
            community_power_line.energy_to_households(self.pv_gen_energy - self.hh_cons + self.v2g_energy_difference)

    def drl_train(self, env):
        """
        In this process, we train the DRL agents
        """
        while self.drl_is_used:
            yield env.timeout(DRL_TRAIN_TIME)  # In each time step we may need to exchange the energy with
            self.drl_agent_v2g1.replay()  # Just call this function to train the agent when we want

    def electricity_cost(self, env):
        """
        Updating the cost of electricity from the file. Representing the real cost of the electricity cost.
        """
        while True:
            yield env.timeout(int(1/self.timestep_size))  # We need to change price cost every hour
            self.el_cost_values = self.el_cost_data[self.el_cost_line]
            self.el_buy_cost = np.round(float(self.el_cost_values.split(',')[4]), 5)
            self.el_sell_cost = np.round(float(self.el_cost_values.split(',')[3]), 5)
            self.el_cost_line += 1

    def track_el_cost(self, env):
        """
        Tracking money cost, depending on the amount of energy the household exchanged with the grid
        """
        while True:  # TO DO: Append it to the list of process in main.py
            yield env.timeout(1)
            # Track the cost, assume that household can sell electricity to the grid as its price
            hh_ener_change = self.pv_gen_energy - self.hh_cons + self.v2g_energy_difference

            if hh_ener_change >= 0:
                self.temp_hh_cost.append(hh_ener_change * self.el_sell_cost)
                self.temp_hh_energy_sell.append(hh_ener_change * self.el_sell_cost)
                self.track_energy_price_household.append(hh_ener_change * self.el_sell_cost)
                self.hh_energy_cost = hh_ener_change * self.el_sell_cost
            else:
                self.temp_hh_cost.append(hh_ener_change * self.el_buy_cost)
                self.temp_hh_energy_buy.append(hh_ener_change * self.el_buy_cost)
                self.track_energy_price_household.append(hh_ener_change * self.el_buy_cost)
                self.hh_energy_cost = hh_ener_change * self.el_buy_cost


            self.track_energy_price_battery.append(self.shared_battery.battery_energy_step_cost)
            self.track_household_energy_price_buy.append(self.el_buy_cost)
            self.track_household_energy_price_sell.append(self.el_sell_cost)
            self.track_shared_bat_price_sell.append(self.shared_battery.sell_energy_price)
            self.track_sharedd_bat_price_buy.append(self.shared_battery.buy_energy_price)

    def collect_simulation_data(self, env):
        """
        Process that will collect, data daily, monthly
        """
        while True:
            yield env.timeout(self.day_time_step_len)
            # restart tracking hour in day
            self.hour_time_step = 0
            self.hour_in_day = 0
            self.daily_energy_col.append(np.sum(self.temp_energy_col))
            self.daily_energy_cons.append(np.sum(self.temp_energy_cons))
            self.daily_energy_cons_v2g1.append(np.sum(self.temp_energy_cons_v2g1))
            self.daily_v2g_energy_deficit.append(np.sum(self.temp_v2g_energy_deficit))
            self.daily_hh_cost.append(np.sum(self.temp_hh_cost))

            self.daily_hh_energy_sell.append(np.sum(self.temp_hh_energy_sell))
            self.daily_hh_energy_buy.append(np.sum(self.temp_hh_energy_buy))
            # empty the temporal lists
            self.temp_energy_col = []
            self.temp_energy_cons = []
            self.temp_energy_cons_v2g1 = []
            self.temp_v2g_energy_deficit = [0.0]
            self.temp_hh_cost = []
            self.temp_hh_energy_sell = [0.0]
            self.temp_hh_energy_buy = [0.0]
            # check if this is end of the month
            self.day_in_a_month += 1
            if self.day_in_a_month == DAYS_IN_MONTH[self.current_month % 12]:
                print("Change over time:", np.sum(self.test_list))
                self.monthly_energy_col.append(sum(self.daily_energy_col))
                self.monthly_energy_cons.append(sum(self.daily_energy_cons))
                self.monthly_energy_cons_v2g1.append(sum(self.daily_energy_cons_v2g1))
                self.monthly_ts_no_energy_v2g1.append(self.v2g1_ts_no_energy)
                self.month_v2g_energy_deficit.append(sum(self.daily_v2g_energy_deficit))
                self.monthly_hh_cost.append(sum(self.daily_hh_cost))
                self.monthly_hh_energy_sell.append(sum(self.daily_hh_energy_sell))
                self.monthly_hh_energy_buy.append(sum(self.daily_hh_energy_buy))
                # empty daily energy collected:
                self.daily_energy_col = []
                self.daily_energy_cons = []
                self.daily_energy_cons_v2g1 = []
                self.daily_v2g_energy_deficit = []
                self.daily_hh_cost = []
                self.daily_hh_energy_sell = []
                self.daily_hh_energy_buy = []
                self.current_month += 1
                self.day_in_a_month = 0
                self.v2g1_ts_no_energy = 0
                self.v2g1_ts_no_energy = 0

                print("Monthly energy collected and consumed:", self.monthly_energy_col,  self.monthly_energy_cons,
                      "Electric vehicles consumption:", self.monthly_energy_cons_v2g1,
                      "Time steps no energy:", self.monthly_ts_no_energy_v2g1,
                      "Energy deficit of a V2G", self.month_v2g_energy_deficit,
                      "Cost of energy in Euros combined:", self.monthly_hh_cost,
                      "Sell in Euros:", self.monthly_hh_energy_sell,
                      "Sell total in Euros:", np.sum(self.monthly_hh_energy_sell),
                      "Buy in Euros:", self.monthly_hh_energy_buy,
                      "Buy total in Euros:", np.sum(self.monthly_hh_energy_buy))
                print("Tracking reward values max:", np.max(self.track_rwd), "Min", np.min(self.track_rwd) , "Average:", np.average(self.track_rwd))
                print("Tracking cost of energy trading of a household per time step min:", np.min(self.track_energy_price_household), "max:", np.max(self.track_energy_price_household), "avg:", np.average(self.track_energy_price_household), "total data points:", len(self.track_energy_price_household), "Removed outliers number of points:", len([x for x in self.track_energy_price_household if 0.1 > x > -.5]))

                print("Tracking cost of energy trading of a shared battery per time step min:", np.min(self.track_energy_price_battery), "max:", np.max(self.track_energy_price_battery), "avg:", np.average(self.track_energy_price_battery), "total data points:", len(self.track_energy_price_battery), "Removed outliers number of points:", len([x for x in self.track_energy_price_battery if 0.2 > (x/2) > -.5]))

            if self.drl_is_used and self.train:
                print("Exploration rate:", self.drl_agent_v2g1.epsilon)
                # Save networks
                self.drl_agent_v2g1.save("drl_trained_nets/" + self.mode + "_total_num_" +
                                         str(self.shared_battery.num_households) + "_household_" + str(self.unique_id)
                                         + "_bat_size_" + str(self.shared_battery.battery_cap) + ".pkl")


def v2g_action(mode, percentage_full=np.nan, env_time=0, hour_in_day=np.nan, change_of_energy=np.nan,
               v2g_charge_energy=np.nan, drl_agent=np.nan, drl_state=np.nan):
    """
    A function that returns a decision on what should the V2G element do.

    :return:
    """
    if mode == MODE_IMMEDIATE: # We charge max if the car is present
        #print("Selected mode of operation is:", mode)
        if percentage_full < ELECTRIC_VEHICLE_FULL_THRESHOLD:
            return CHARGE_V2G
        else:
            return NO_CHANGE_V2G
    elif mode == MODE_ALTER:
        #print("Selected mode of operation is:", mode)
        return env_time % NUM_ACTIONS # Note we make a slight mistake as we skip certain time-steps DO NOT USE IN PAPER!
    elif mode == MODE_NIGHT:

        if hour_in_day in NIGHT_MODE_HOURS:  # We are at night when we allow to charge
            if percentage_full < ELECTRIC_VEHICLE_FULL_THRESHOLD:
                #print("Night mode input:", percentage_full, hour_in_day, change_of_energy, v2g_charge_energy)
                return CHARGE_V2G
            else:
                return NO_CHANGE_V2G

    elif mode == MODE_DRL_SINGLE:
        return drl_agent.act(drl_state)
    elif mode == MODE_COMAD:
        return drl_agent.act(drl_state)
    else:
        print("ERROR: Selected mode of operation is INVALID")


def rwd_drl(v2g_percentage, energy_exchange, energy_trade): #), household_diff, v2gcons):
    """
    We determine the reward for the DRL agents. Based on the energy in V2G and exchanged energy. In case of a single
    agent, we take energy the household exchanged with others, i.e., consider the household being by itself. In the
    multi-agent approach, we assume that household is part of the community thus we use energy community exchanged with
    external grid.
    :param v2g_percentage: amount of energy available in the V2G element
    :param energy_exchange: amount of energy the system traded in a step
    :param energy_trade: cost in Euros for the traded energy
    :return: return reward signal
    """

    rwd = 0
    if DRL_RWD_MIN_BAT <= v2g_percentage <= DRL_RWD_MAX_BAT:
        rwd = 1.0 * (1 + v2g_percentage / (DRL_RWD_MAX_BAT - DRL_RWD_MIN_BAT))
    else:
        rwd = 1.0 * (v2g_percentage - 1)

    rwd += 10*np.clip(energy_exchange, -REWARD_CONS_RANGE, 0)/REWARD_CONS_RANGE

    if energy_trade >= 0:
        rwd += 10 * np.clip(energy_trade, 0, DRL_RWD_COST_MAX_NORM) / DRL_RWD_COST_MAX_NORM
    else:
        rwd -= 10 * np.clip(energy_trade, DRL_RWD_COST_MIN_NORM, 0) / DRL_RWD_COST_MIN_NORM

    #print("Reward function input:", v2g_percentage, energy_exchange, energy_trade, "Determined Reward:", rwd)
    return rwd


def rwd_drl_single(v2g_percentage, energy_exchange):
    """
    We determine the reward for the DRL agents. Based on the energy in V2G and exchanged energy. In case of a single
    agent, we take energy the household exchanged with others, i.e., consider the household being by itself. In the
    multi-agent approach, we assume that household is part of the community thus we use energy community exchanged with
    external grid.
    :param v2g_percentage: amount of energy available in the V2G element
    :param energy_exchange: amount of energy the system traded in a step
    :param energy_trade: cost in Euros for the traded energy
    :return: return reward signal
    """

    rwd = 0
    if DRL_RWD_MIN_BAT <= v2g_percentage <= DRL_RWD_MAX_BAT:
        rwd = 1.0 * (1 + v2g_percentage / (DRL_RWD_MAX_BAT - DRL_RWD_MIN_BAT))
    else:
        rwd = 1.0 * (v2g_percentage - 1)

    rwd += 10*np.clip(energy_exchange, -REWARD_CONS_RANGE, 0)/REWARD_CONS_RANGE

    return rwd

