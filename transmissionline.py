import simpy


class TransmissionLine(object):
    """With this class we represent power line to exchange the energy between households and shared battery."""
    def __init__(self, env, num_households):
        self.env = env
        self.num_households = num_households
        self.transmission_line = simpy.Store(env)  # We create object to share amount of energy each household needs
        self.req_energy = simpy.Resource(env, capacity=self.num_households)  # Creating a resource to share energy

    def energy_to_households(self, energy):
        """Households use it to inform the Shared battery how much energy to compensate.
           Negative energy means households needs energy, positive is a surplus of energy. """
        self.transmission_line.put(energy)

    def energy_from_households(self):
        """Shared battery uses it to receive information how much energy to compensate or store from households."""
        return self.transmission_line.get()