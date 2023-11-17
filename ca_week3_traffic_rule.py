import numpy as np
import itertools
import matplotlib.pyplot as plt

from pyics import Model


def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.  

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""
    result = []
    
    while n > 0:
        remainder = n % k
        result.insert(0, remainder)
        n //= k

    return result



class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('lambda_param', 0.1) #for langtons lamda parameter
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('rule', 30, setter=self.setter_rule)
        self.make_param('rule_table_method', 'table_walk_through', setter=self.setter_rule_table_method) # methods

        self.config = np.zeros([self.height, self.width], dtype=int)

        self.build_rule_set()

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))
    
    # langtons implementation start

    def setter_rule_table_method(self, val):
        methods = ['table_walk_through', 'random_table']
        return val if val in methods else 'table_walk_through'

    def langton_to_rule(self, langton_lambda):
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return int(langton_lambda * max_rule_number)

    def rule_to_langton(self, rule_number):
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return rule_number / max_rule_number

    def set_langton_lambda(self, langton_lambda):
        rule_number = self.langton_to_rule(langton_lambda)
        self.rule = rule_number
        self.build_rule_set()

    def set_rule_table_method(self, method):
        self.rule_table_method = method
        self.build_rule_set()

    def calculate_entropy(self, config):
        flattened_config = config.flatten()
        unique_states, state_counts = np.unique(flattened_config, return_counts=True)
        probabilities = state_counts / len(flattened_config)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    # langtons functions end

    def build_rule_set(self):
        if self.rule_table_method == 'table_walk_through':
            rules = []
            max_rule = self.k ** (2 * self.r + 1)
            while len(rules) < max_rule:
                rules.append(0)

            base_k = decimal_to_base_k(self.rule, self.k)
            rules[-len(base_k):] = base_k

            states = list(range(self.k))
            beginning_states = reversed(list(itertools.product(states, repeat=2 * self.r + 1)))
            rule_set = {}

            for i, beginning_state in enumerate(beginning_states):
                rule_set[beginning_state] = rules[i]

            self.rule_set = rule_set

        elif self.rule_table_method == 'random_table':
            # random-table method
            rule_set_size = self.k ** (2 * self.r + 1)
            max_rule_number = self.k ** rule_set_size

            random_rule = np.random.randint(0, max_rule_number)
            base_k = decimal_to_base_k(random_rule, self.k)

            rule_set = {}
            states = list(range(self.k))
            beginning_states = reversed(list(itertools.product(states, repeat=2 * self.r + 1)))

            for i, beginning_state in enumerate(beginning_states):
                if i < len(base_k):
                    rule_set[beginning_state] = base_k[i]
                else:
                    rule_set[beginning_state] = 0

            self.rule_set = rule_set

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""
    
        return self.rule_set.get(tuple(inp), 0)



    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""
        initial_row = np.random.randint(0, self.k, size=self.width)
        #initial_row = np.zeros(self.width, dtype=int)
        #initial_row[self.width // 2] = 1


        return initial_row

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.height:
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [i % self.width
                    for i in range(patch - self.r, patch + self.r + 1)]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)

def generate_rule_tables_from_lambda(lambda_values, num_rules, num_iterations, num_steps, width, method='table_walk_through'):
    rule_tables = {}

    for langton_lambda in lambda_values:
        rule_tables[langton_lambda] = find_average_cycle_length(
            num_rules, num_iterations, num_steps, width, langton_lambda=langton_lambda, method=method
        )

    return rule_tables

def find_average_cycle_length(num_rules, num_iterations, num_steps, width, langton_lambda=None, method='table_walk_through'):
    average_length_dic = {}

    for rule in range(num_rules + 1):
        cycle_lengths = []

        for iteration in range(num_iterations):
            sim = CASim()
            sim.height = num_steps
            sim.width = width
            sim.reset()

            if langton_lambda is not None:
                sim.set_langton_lambda(langton_lambda)
                sim.set_rule_table_method(method)

            sim.rule = rule
            length = 0

            initial_state = sim.config[sim.t].copy()

            while length < num_steps:
                if length > 0 and np.array_equal(sim.config[length], initial_state):
                    cycle_lengths.append(length)
                    break

                sim.step()

                length += 1

            if length == num_steps:
                cycle_lengths.append(length)

        average_length_dic[rule] = cycle_lengths

    return average_length_dic

class Rule184CA(CASim):
    def __init__(self, car_density, height, width):
        super().__init__()
        self.car_density = car_density
        self.height = height
        self.width = width

    def setup_initial_row(self):
        initial_row = np.random.choice([0, 1], size=self.width, p=[1 - self.car_density, self.car_density])
        return initial_row

    def check_rule(self,inp):
        left, center, right = inp
        rules = {
            (1, 1, 1): 1,
            (1, 1, 0): 0,
            (1, 0, 1): 1,
            (1, 0, 0): 1,
            (0, 1, 1): 1,
            (0, 1, 0): 0,
            (0, 0, 1): 0,
            (0, 0, 0): 0
        }
        return rules[(left, center, right)]
    

#"""
if __name__ == '__main__':
    num_steps = 50
    ca_size = 50
    car_densities = [0.4, 0.9]

    for density in car_densities:
        ca_sim = Rule184CA(car_density=density)
        ca_sim.width = ca_size
        ca_sim.height = num_steps
        ca_sim.reset()

        for step in range(0, num_steps):
            ca_sim.step()

        plt.figure(figsize=(6, 6))
        plt.imshow(ca_sim.config, cmap='binary', aspect='auto')
        plt.title(f"CA evolution at step 50 (density: {density})")
        plt.show()
#"""

def calculate_car_flow(sim, density, num_iterations, num_steps, n):
    avg_flow_per_time = 0
    sim.car_density = density
    sim.width = n
    sim.height = num_steps
    
    for iteration in range(num_iterations):
        sim.reset()

    
        flow = 0
        for step in range(num_steps):
            sim.step()
            flow += np.sum(sim.config[:, -1])    

        avg_flow_per_time += flow / num_steps
    
    avg_flow_among_iterations = avg_flow_per_time / num_iterations

    
    return avg_flow_among_iterations


def measure_car_flow(sim, densities, num_iterations, num_steps, n):
    avg_flows = []
    for density in densities:
        car_flow = calculate_car_flow(sim, density, num_iterations, num_steps, n)
        avg_flows.append(car_flow)
    return avg_flows

if __name__ == '__main__':
    t = 1000
    n = 50
    r = 10
 
    densities = np.linspace(0.0, 1.0, 40)

    ca_sim = Rule184CA(car_density=0.1, height=t, width=n)

    avg_flows = measure_car_flow(ca_sim, densities, r, t, n)

    plt.plot(densities, avg_flows, marker='o')
    plt.title('Average car flow vs initial density')
    plt.xlabel('Initial density of cars')
    plt.ylabel('Average car flow per unit time')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    t = 5
    n = 50
    r = 3
 
    densities = np.linspace(0.0, 1.0, 40)

    ca_sim = Rule184CA(car_density=0.1, height=t, width=n)

    avg_flows = measure_car_flow(ca_sim, densities, r, t, n)

    plt.plot(densities, avg_flows, marker='o')
    plt.title('Average car flow vs initial density')
    plt.xlabel('Initial density of cars')
    plt.ylabel('Average car flow per unit time')
    plt.grid(True)
    plt.show()

