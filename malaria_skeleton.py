import matplotlib.pyplot as plt
import numpy as np
import malaria_visualize


class Model:
    def __init__(self, params):
        """
        Model parameters
        Initialize the model with the width and height parameters.
        """
        self.height = params.get('height', 50)
        self.width = params.get('width', 50)
        self.nHuman = params.get('nHuman', 400)
        self.nMosquito = params.get('nMosquito', 200)
        self.humanInfectionProb = params.get('humanInfectionProb', 0.25)
        self.mosquitoInfectionProb = params.get('mosquitoInfectionProb', 0.9)
        self.biteProb = params.get('biteProb', 1.0)
        

        self.mosquitoNetCoverage = params.get('mosquito_net_coverage', 0.8)
        self.antimalarialDrugCoverage = params.get('antimalarial_drug_coverage', 0.4)
        self.antimalarialDrugEffectiveness = params.get('antimalarial_drug_effectiveness', 0.8)

        vaccinationCoverage = params.get('vaccinationCoverage', 0) # Vaccination coverage parameter
        initMosquitoHungry = params.get('initMosquitoHungry', 0.5)
        initHumanInfected = params.get('initHumanInfected', 0.2)

        # etc.

        """
        Data parameters
        To record the evolution of the model
        """
        self.infectedCount = 0
        self.deathCount = 0
        self.bitesThroughNet = 0
        # etc.

        """
        Population setters
        Make a data structure in this case a list with the humans and mosquitos.
        """
        self.humanPopulation = self.set_human_population(initHumanInfected, vaccinationCoverage)
        self.mosquitoPopulation = self.set_mosquito_population(initMosquitoHungry)

    def set_human_population(self, initHumanInfected, vaccinationCoverage):
        """
        This function makes the initial human population, by iteratively adding
        an object of the Human class to the humanPopulation list.
        The position of each Human object is randomized. A number of Human
        objects is initialized with the "infected" state.
        """
        humanPopulation = []
        occupied_positions = set()  # To track occupied positions

        for i in range(self.nHuman):
            # Keep generating positions until a non-overlapping one is found
            while True:
                x = np.random.randint(self.width)
                y = np.random.randint(self.height)
                # Check if the position is already occupied
                if (x, y) not in occupied_positions:
                    break  # Break the loop if position is not occupied

            occupied_positions.add((x, y))  # Add position 

            if (i / self.nHuman) <= initHumanInfected:
                state = 'I'  # I for infected
            else:
                # Check if the human is immune based on vaccination coverage
                if np.random.uniform() <= vaccinationCoverage:
                    state = 'R'  # R for immune
                else:
                    state = 'S'  # S for susceptible
            humanPopulation.append(Human(x, y, state))

        return humanPopulation


    def set_mosquito_population(self, initMosquitoHungry):
        """
        This function makes the initial mosquito population, by iteratively
        adding an object of the Mosquito class to the mosquitoPopulation list.
        The position of each Mosquito object is randomized.
        A number of Mosquito objects is initialized with the "hungry" state.
        """
        mosquitoPopulation = []
        for i in range(self.nMosquito):
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)
            if (i / self.nMosquito) <= initMosquitoHungry:
                hungry = True
            else:
                hungry = False
            mosquitoPopulation.append(Mosquito(x, y, hungry))
        return mosquitoPopulation

    def update(self):
        """
        Perform one timestep:
        1.  Update mosquito population. Move the mosquitos. If a mosquito is
            hungry it can bite a human with a probability biteProb.
            Update the hungry state of the mosquitos.
        2.  Update the human population. If a human dies remove it from the
            population, and add a replacement human.
        """
        for i, m in enumerate(self.mosquitoPopulation):
            m.move(self.height, self.width)
            for h in self.humanPopulation:
                if m.position == h.position and m.hungry and np.random.uniform() <= self.biteProb:
                    # Check if the human is under a mosquito net
                    if h.is_under_mosquito_net(self.mosquitoNetCoverage):
                        # Reduce the probability of bite if under a net
                        if np.random.uniform() <= self.biteProb * 0.1:  # Hypothetical reduction rate
                            m.bite(h, self.humanInfectionProb, self.mosquitoInfectionProb)
                            self.bitesThroughNet += 1  # Increment bites through net counter
                    else:
                        m.bite(h, self.humanInfectionProb, self.mosquitoInfectionProb)
            m.update_hunger_state()

        new_infected_count = 0
        new_death_count = 0

        for j, h in enumerate(self.humanPopulation):

            # Check if a human gets infected and update their state
            if h.state == 'S':
                for m in self.mosquitoPopulation:
                    if m.position == h.position and m.infected and np.random.uniform() <= self.humanInfectionProb:
                        # Check if the human is taking antimalarial drugs
                        if np.random.uniform() <= self.antimalarialDrugCoverage:
                            # Reduce the probability of infection with the drug's effectiveness
                            if np.random.uniform() <= self.antimalarialDrugEffectiveness:
                                h.state = 'I'
                                new_infected_count += 1  # Increment infected count
                                break  # Stop checking for infections after the first infected mosquito
                        else:
                            h.state = 'I'
                            new_infected_count += 1  # Increment infected count
                            break  # Stop checking for infections after the first infected mosquito
            elif h.state == 'I':
                # Check if a human gets treated and update their state
                if np.random.uniform() <= 0.01:
                    
                    if np.random.uniform() <= 0.5:
                        h.state = 'R'  # R for immune
                    else:
                        h.state = 'S'  # Successfully treated, move to susceptible state
                    new_infected_count -= 1  # Increment treated count
                
                elif np.random.uniform() <= 0.02:
                    del self.humanPopulation[j]
                    new_death_count += 1
                    x = np.random.randint(self.width)
                    y = np.random.randint(self.height)
                    state = 'S'  # New human starts as susceptible
                    self.humanPopulation.append(Human(x, y, state))
                
            
            # Check if a human dies based on random probability
            if np.random.uniform() < 0.0001:
                # Remove the dead human from the population
                del self.humanPopulation[j]
                new_death_count += 1  # Increment death count
                # Add a replacement human at a random position
                x = np.random.randint(self.width)
                y = np.random.randint(self.height)
                state = 'S'  # New human starts as susceptible
                self.humanPopulation.append(Human(x, y, state))
            
        # Update the total counts
        self.infectedCount += new_infected_count
        self.deathCount += new_death_count

        return self.infectedCount, self.deathCount


class Mosquito:
    def __init__(self, x, y, hungry, time_to_become_hungry = 1) :
        """
        Class to model the mosquitos. Each mosquito is initialized with a random
        position on the grid. Mosquitos can start out hungry or not hungry.
        All mosquitos are initialized infection free (this can be modified).
        """
        self.position = [x, y]
        self.hungry = hungry
        self.infected = False
        self.time_to_become_hungry = time_to_become_hungry  # Time steps to become hungry
        self.time_since_last_bite = 0  # Counter for tracking time since last bite

    def update_hunger_state(self):
        """
        Update the hunger state of the mosquito based on time steps passed.
        """
        self.time_since_last_bite += 1
        if self.time_since_last_bite >= self.time_to_become_hungry:
            self.hungry = True
            self.time_since_last_bite = 0  # Reset the counter after becoming hungry

    def bite(self, human, humanInfectionProb, mosquitoInfectionProb):
        """
        Function that handles the biting. If the mosquito is infected and the
        target human is susceptible, the human can be infected.
        If the mosquito is not infected and the target human is infected, the
        mosquito can be infected.
        After a mosquito bites it is no longer hungry.
        """
        if self.infected and human.state == 'S':
            if np.random.uniform() <= humanInfectionProb:
                human.state = 'I'
        elif not self.infected and human.state == 'I':
            if np.random.uniform() <= mosquitoInfectionProb:
                self.infected = True
        self.hungry = False

    def move(self, height, width):
        """
        Moves the mosquito one step in a random direction.
        """
        deltaX = np.random.randint(-1, 2)
        deltaY = np.random.randint(-1, 2)
        """
        The mosquitos may not leave the grid. There are two options:
                      - fixed boundaries: if the mosquito wants to move off the
                        grid choose a new valid move.
                      - periodic boundaries: implement a wrap around i.e. if
                        y+deltaY > ymax -> y = 0. This is the option currently implemented.
        """
        self.position[0] = (self.position[0] + deltaX) % width
        self.position[1] = (self.position[1] + deltaY) % height


class Human:
    def __init__(self, x, y, state):
        """
        Class to model the humans. Each human is initialized with a random
        position on the grid. Humans can start out susceptible or infected
        (or immune).
        """
        self.position = [x, y]
        self.state = state

    def is_under_mosquito_net(self, mosquito_net_coverage):
        """
        Check if the human is under a mosquito net based on the coverage probability.
        """
        return np.random.uniform() <= mosquito_net_coverage


if __name__ == '__main__':
    """
    Simulation parameters
    """
    simulation_params = {
        'width': 50,
        'height': 50,
        'nHuman': 672,
        'nMosquito': 1000,
        'initMosquitoHungry': 0.5,
        'initHumanInfected': 0.001,
        'humanInfectionProb': 0.25,
        'mosquitoInfectionProb': 0.9,   
        'biteProb': 1,
        'mosquito_net_coverage': 0.8,
        'antimalarial_drug_coverage': 0.4,
        'antimalarial_drug_effectiveness': 0.8,
        'vaccinationCoverage': 0

        # parameters here
    }

    fileName = 'malaria_simulation'
    timeSteps = 100
    t = 0
    plotData = False
    """
    Run a simulation for an indicated number of timesteps.
    """
    file = open(fileName + '.csv', 'w')
    sim = Model(simulation_params)
    vis = malaria_visualize.Visualization(sim.height, sim.width)

    print('Starting simulation')
    while t < timeSteps:
        [d1, d2] = sim.update()  # Catch the data
        line = f"{t},{d1},{d2},{sim.bitesThroughNet}\n"  # Separate the data with commas
        file.write(line)  # Write the data to a .csv file
        vis.update(t, sim.mosquitoPopulation, sim.humanPopulation)
        t += 1

    file.close()
    vis.persist()

    if plotData:
        """
        Make a plot by from the stored simulation data.
        """
        data = np.loadtxt(fileName+'.csv', delimiter=',')
        time = data[:, 0]
        infectedCount = data[:, 1]
        deathCount = data[:, 2]
        plt.figure()
        plt.plot(time, infectedCount, label='infected')
        plt.plot(time, deathCount, label='deaths')
        plt.legend()
        plt.show()
