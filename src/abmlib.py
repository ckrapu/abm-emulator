import enum
import numpy as np

from mesa         import Agent, Model
from mesa.time    import RandomActivation
from mesa.space   import MultiGrid

from mesa.datacollection import DataCollector

from tqdm import tqdm

class InfectionModel(Model):
    """A model for infection spread."""

    def __init__(self, N=10, width=10, height=10, ptrans=0.1,
                 death_rate=0.02, recovery_days=21,
                 recovery_sd=7,p_infected_initial_log10=-3,
                 pdoctor_log10=-3, pcure=0.80):

        self.num_agents = N
        self.recovery_days = recovery_days
        self.recovery_sd = recovery_sd
        self.ptrans = ptrans
        self.pdoctor = 10**pdoctor_log10
        self.pcure = pcure
        self.death_rate = death_rate
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True
        self.dead_agents = []

        p_infected_initial = 10**p_infected_initial_log10

        # Create agents
        for i in range(self.num_agents):
            a = MyAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

            #make some agents infected at start
            infected = np.random.choice([0,1], p=[1-p_infected_initial,p_infected_initial])
            if infected == 1:
                a.state = State.INFECTED
                a.recovery_time = self.get_recovery_time()

        self.datacollector = DataCollector(          
            agent_reporters={"State": "state"})

    def get_recovery_time(self):
        return int(self.random.normalvariate(self.recovery_days,self.recovery_sd))

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        
class State(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2

class MyAgent(Agent):
    """ An agent in an epidemic model."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = State.SUSCEPTIBLE  
        self.infection_time = 0
        self.doctor = np.random.choice([0,1], p=[1-model.pdoctor, model.pdoctor])

    def move(self):
        """Move the agent"""

        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)

        n_possible = len(possible_steps)

        if self.doctor:
            infected_counts = np.asarray([np.sum([x.state == State.INFECTED for x in self.model.grid.get_cell_list_contents([location])]) for location in possible_steps])
            if any(infected_counts):
                weights = infected_counts / np.sum(infected_counts)
            else: # Doctor doesn't move if no patients
                return

            new_position_idx = np.random.choice(np.arange(n_possible), p=weights)

        else:
            new_position_idx = np.random.choice(np.arange(n_possible))

        new_position = possible_steps[new_position_idx]

        self.model.grid.move_agent(self, new_position)

    def status(self):
        """Check infection status"""

        if self.state == State.INFECTED:     
            drate = self.model.death_rate
            alive = np.random.choice([0,1], p=[drate,1-drate])

            if not alive:
                self.state = State.REMOVED
                self.model.schedule.remove(self)            
            t = self.model.schedule.time-self.infection_time

            if t >= self.recovery_time:          
                self.state = State.REMOVED

    def contact(self):
        """Find close contacts and infect or be cured by a doctor"""
        cellmates = self.model.grid.get_cell_list_contents([self.pos])       
        if len(cellmates) > 1:
            for other in cellmates:

                if other.doctor and self.state == State.INFECTED and np.random.choice([0, 1], p=[1-self.model.pcure, self.model.pcure]):
                    self.state = State.REMOVED

                if self.random.random() > self.model.ptrans:
                    continue

                if self.state is State.INFECTED and other.state is State.SUSCEPTIBLE:                    
                    other.state = State.INFECTED
                    other.infection_time = self.model.schedule.time
                    other.recovery_time = self.model.get_recovery_time()

    def step(self):
        self.status()
        self.move()
        self.contact()

state_dict = {
    0:'Susceptible',
    1:'Infected',
    2:'Removed',
    }

def SIR(steps, model_kwargs={}):
    
    model  = InfectionModel(**model_kwargs)
    grid_state       = np.zeros((steps, model.grid.width, model.grid.height, len(State)))
    doctor_locations = np.zeros((steps, model.grid.width, model.grid.height))
    for i in tqdm(range(steps)):
        model.step()

        for cell in model.grid.coord_iter():
            agents, x, y = cell

            for a in agents:
                grid_state[i, x,y,a.state] +=1
                doctor_locations[i, x, y]  += a.doctor
                
                
    return model, grid_state, doctor_locations