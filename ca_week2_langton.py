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

    def setter_rule(self, val):
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))

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
                rule_set[beginning_state] = base_k[i]

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
    average_length_dic = {}  # Use a dictionary to store results for each rule

    for rule in range(num_rules + 1):
        cycle_lengths = []

        for iteration in range(num_iterations):
            sim = CASim()
            sim.height = num_steps
            sim.width = width  # Set width here
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

if __name__ == '__main__':
    num_rules = 256
    num_iterations = 10
    num_steps = 100
    width = 10

    lambda_values = [0.1, 0.3, 0.5]
    rule_table_method = 'table_walk_through'  # or 'random_table'

    average_lengths = find_average_cycle_length(num_rules, num_iterations, num_steps, width)
    
    rule_tables = generate_rule_tables_from_lambda(lambda_values, num_rules, num_iterations, num_steps, width, method=rule_table_method)

    for langton_lambda, average_lengths in rule_tables.items():
        print(f"Langton lambda: {langton_lambda}")
        print(f"Rule table method: {rule_table_method}")
        print("Rule\tAverage steps\tLowest steps")
        for rule, cycle_lengths in average_lengths.items():
            avg_steps = sum(cycle_lengths) / num_iterations
            lowest_steps = min(cycle_lengths)

            # entropy calculation
            
            sim = CASim()
            sim.height = num_steps
            sim.width = width
            sim.reset()
            sim.set_langton_lambda(langton_lambda)
            sim.set_rule_table_method(rule_table_method)
            sim.rule = rule
            
            entropy = sim.calculate_entropy(sim.config)

            print(f"{rule}\t{avg_steps:.2f}\t{lowest_steps}\t{entropy:.4f}")

        x_values = []
        y_values = []

        for rule, cycle_lengths in average_lengths.items():
            x_values.extend([rule] * len(cycle_lengths))
            y_values.extend(cycle_lengths)

        plt.scatter(x_values, y_values, marker="x", s=6, label=f"Lambda={langton_lambda}", alpha=0.7)

    plt.xlabel('Rule number')
    plt.ylabel(f'Steps')
    plt.title(f'CA cycle length of {num_rules} rules tested within {num_steps} steps, width = {width}')
    plt.legend()
    plt.grid()
    plt.show()
