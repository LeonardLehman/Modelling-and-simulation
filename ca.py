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
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('rule', 30, setter=self.setter_rule)

        self.config = np.zeros([self.height, self.width], dtype=int)

        self.build_rule_set()

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""
        
        
        rules = []
        max_rule = self.k ** (2 * self.r + 1)

        while len(rules) < max_rule:
            rules.append(0)

        base_k = decimal_to_base_k(self.rule, self.k)
        rules[-len(base_k):] = base_k

        
        states = list(range(self.k))
        beginning_states = reversed(list(itertools.product(states, repeat=2*self.r + 1)))
        rule_set = {}

        for i, beginning_state in enumerate(beginning_states):
            rule_set[beginning_state] = rules[i]
        
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

def find_average_cycle_length(num_rules, num_iterations, num_steps):
    average_lengths = []

    for rule in range(num_rules + 1):
        cycle_lengths = []

        for iteration in range(num_iterations):
            sim = CASim()
            sim.height = num_steps
            sim.width = 10
            sim.reset()
            sim.rule = rule
            length = 0

            initial_state = sim.config[sim.t].copy()  # Track the initial state

            while length < num_steps:
                if length > 0 and np.array_equal(sim.config[length], initial_state):
                    cycle_lengths.append(length)
                    break

                sim.step()
                    
                length += 1

            # If no cycle was detected, use the entire simulation length
            if length == num_steps:
                cycle_lengths.append(length)

        average_length = sum(cycle_lengths) / num_iterations
        average_lengths.append(average_length)

    return average_lengths

if __name__ == '__main__':
    num_rules = 256
    num_iterations = 100
    num_steps = 100

    average_lengths = find_average_cycle_length(num_rules, num_iterations, num_steps)

    for rule, avg_length in enumerate(average_lengths):
        print(f"Rule {rule}: Average Length Until Cycle = {avg_length}")

    rules = list(range(num_rules + 1))

    plt.bar(rules, average_lengths, align='center', alpha=0.7)
    plt.xlabel('Rule number')
    plt.ylabel(f'Average steps of {num_iterations} iterations')
    plt.title(f'Average CA cycle lenght of {num_rules} rules tested within {num_steps} steps')
    plt.grid()
    plt.show()