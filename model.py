import mesa
import numpy as np
import pandas as pd
from operator import add
from space import SingleGrid
from scipy.stats import truncnorm
from scipy.ndimage import convolve

import warnings
warnings.simplefilter("ignore", UserWarning)
from segregation.multigroup import MultiDissim
from segregation.singlegroup import SpatialDissim

class SchellingAgent(mesa.Agent):
    
    def __init__(self, unique_id, model, agent_type, homophily):
        """
        Create a new Schelling agent.

        Args:
           unique_id: Unique identifier for the agent.
           model: Model instance
           agent_type: integer for the agent's type
           homophily: fraction similar neighbours required
        """
        super().__init__(unique_id, model)
        
        self.satisfied = False
        self.type = agent_type
        self.homophily = homophily

    def step(self):
        """
        Steps the agent if allowed to move and unsatisfied.
        """
        
        # Check if the agent is allowed to move
        if self.model.agents_stepped < self.model.max_moves:

            # Only move if not satisfied
            if not self.satisfied:

                # Move but save the old position
                old_x, old_y = self.pos
                self.model.grid.move_to_empty(self)
                new_x, new_y = self.pos

                # Switch attrs
                self.model.household_attrs[old_x, old_y] = 0
                self.model.household_attrs[new_x, new_y] = self.type + 1

    def calc_similar(self):
        """
        Calculates the number of similar neighbours.

        Returns:
            count (int): total agents in the neighbourhood
            similar (int): number of similar agents in the
                neighbourhood
        """
        x, y = self.pos
        counts = self.model.compositions[:, x, y]
        total = counts[1:].sum() 
        similar = counts[self.type + 1]

        # Otherwise zerodivision.
        if total==0:
            total = 1

        return total, similar

class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(self, pars={
                    'width':80, 
                    'height':80, 
                    'density':0.9,
                    'max_steps':100, 
                    'mode':'Heterogeneous',
                    'minority_pc':0.5, 
                    'window_size':30, 
                    'conv_threshold':0.01,
                    'radius':1, 
                    'torus':True,
                    'mu1':.3, 
                    'std1':.1,
                    'mu2':.4, 
                    'std2':.05,
                    'move_fraction':0.15,
                    'filename':'test.npz'
                    }):
        """ """
        
        # Set all the parameter values
        self.mode = pars['mode']
        self.torus = pars['torus']
        self.width = pars['width']
        self.height = pars['height']
        self.radius = pars['radius']
        self.density = pars['density']
        self.filename = pars['filename']
        self.max_steps = pars['max_steps']
        self.minority_pc = pars['minority_pc']
        self.window_size = pars['window_size']
        self.move_fraction = pars['move_fraction']
        self.conv_threshold = pars['conv_threshold']

        self.size = self.width * self.height
        self.max_moves = int(self.move_fraction * self.density * self.size)

        self.mu1 = pars['mu1']
        self.std1 = pars['std1']
        self.mu2 = pars['mu2']
        self.std2 = pars['std2']
        
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = SingleGrid(self.width, self.height, torus=self.torus)

        self.num_satisfied = 0
        self.avg_fraction_sim = 0
        self.fraction_sat_1 = 0
        self.fraction_sat_2 = 0
        self.satisfied_fraction = 0
        self.convergence_metric = []
        self.household_attrs = np.zeros((self.width, self.height), dtype='int8')

        # Correct for the density in the agent types (0 is empty)
        if type(self.minority_pc)==float:
            type_probs = self.density*np.array([self.minority_pc, 1-self.minority_pc])
        else: 
            type_probs = self.density*np.array(self.minority_pc)

        size = (self.width, self.height)
        type_probs = np.insert(type_probs, 0, 1-self.density)
        types = np.random.choice(a=type_probs.shape[0], size=size, p=type_probs)

        index = 0
        num_agents_1, num_agents_2 = 0, 0

        # Check if there's a standard deviation
        if self.std1 > 0:
            self.homophilies_g1 = truncnorm.rvs(
                    (0 - self.mu1) / self.std1, (1 - self.mu1) / self.std1, 
                    loc=self.mu1, scale=self.std1, size=self.size
                    )
        else:
            self.homophilies_g1 = [self.mu1]*self.size

        # Sample according to the group specific parameters
        if self.mode.lower()=='heterogeneous':

            if self.std1 > 0:
                self.homophilies_g2 = truncnorm.rvs(
                        (0 - self.mu2) / self.std2, (1 - self.mu2) / self.std2, 
                        loc=self.mu2, scale=self.std2, size=self.size
                        )
            else:
                self.homophilies_g2 = [self.mu2]*self.size

        # If homogeneous: homophilies_g1==homophilies_g2
        else:
            self.homophilies_g2 = self.homophilies_g1.copy()

        for pos, agent_type in np.ndenumerate(types):
            
            if agent_type == 0:
                continue
            elif agent_type == 1:
                agent_homophily = self.homophilies_g1[index]
                num_agents_1 += 1
            elif agent_type == 2:
                agent_homophily = self.homophilies_g2[index]
                num_agents_2 += 1

            agent = SchellingAgent(unique_id=index, 
                                model=self, 
                                agent_type=agent_type-1, 
                                homophily=agent_homophily)
            self.grid.position_agent(agent, pos)
            self.schedule.add(agent)
            x, y = agent.pos
            self.household_attrs[x, y] = agent_type

            index += 1

        self.running = True
        self.num_agents = index
        self.num_agents_1 = num_agents_1
        self.num_agents_2 = num_agents_2

        # Update calculations and collect data      
        self.collect_data()

    def step(self):
        """
        Run one step of the model. 
        """
        self.num_satisfied = 0  # Reset counter of happy agents
        self.agents_stepped = 0 # Reset counter of agents stepped
        self.schedule.step()
        self.collect_data()

    def calculate_compositions(self):
        """
        Updates all local residential compositions assuming households have
        the SAME RADIUS.
        """

        # Should it wrap around the edges or not?
        if self.torus:
            mode = "wrap"
        else:
            mode = "constant"

        radius = self.radius
        dim = radius * 2 + 1
        kernel = np.ones((dim, dim), dtype=int)
        kernel[radius, radius] = 0
        
        counts = [0, 0, 0]
        for group in [0, 1, 2]:
            counts[group] = convolve(
                (self.household_attrs==group).astype(int), 
                kernel, mode=mode, cval=0)
        self.compositions = np.array(counts)

    def collect_data(self):
        
        self.calculate_compositions()

        # Update agent attributes
        num_satisfied = 0
        sat_1, sat_2 = 0, 0
        fractions = np.zeros(self.num_agents)
        for i, agent in enumerate(self.schedule.agents):

            total, similar = agent.calc_similar()
            fraction_sim = similar / total
            agent.fraction_sim = fraction_sim
            fractions[i] = fraction_sim

            if fraction_sim < agent.homophily:
                agent.satisfied = 0
            else:
                agent.satisfied = 1
                if agent.type==0:
                    sat_1 += 1
                elif agent.type==1:
                    sat_2 += 1

                num_satisfied += 1

        self.num_satisfied = num_satisfied
        fraction = num_satisfied / self.num_agents
        self.satisfied_fraction = fraction
        self.convergence_metric.append(fraction)
        self.avg_fraction_sim = fractions.mean()
        self.fraction_sat_1 = sat_1 / self.num_agents_1
        self.fraction_sat_2 = sat_2 / self.num_agents_2

    def convergence_check(self):

        if self.num_satisfied == self.num_agents:
            return True

        metric = self.convergence_metric
        if len(metric) >= self.window_size:
            metric_arr = np.array(metric[-self.window_size:])
            mad = np.abs(metric_arr - metric_arr.mean())
            if np.all(mad < self.conv_threshold):
                return True

        return False

    def calc_neighbourhood_compositions(self, n=8):
        attrs = self.household_attrs
        maximum = attrs.max()
        compositions = [0]*(n*n)
        i = 0
        for h in range(n):
            for v in range(n):
                neighbourhood = attrs[h * n : h * n + n, v * n : v * n + n]
                counts = np.bincount(neighbourhood.flatten(), minlength=maximum)
                compositions[i] = list(counts)[1:] # discard empties
                i += 1

        return compositions

    def calculate_segregation(self):
        """
        Only for visualisation purposes!
        """
        size = self.width
        sizes = (2, 4, 8, 16)
        if size != 80:
            return pd.DataFrame(columns=[str(k) for k in sizes])

        attrs = self.household_attrs
        maximum = attrs.max()
        cols = ['group' + str(i) for i in range(1, maximum+1)]
        segregation = []
        
        for i in sizes:

            neighbourhoods = pd.DataFrame(columns=cols, index=range(i*i))
            j = int(size / i)
            start_x, start_y, index = 0, 0, 0

            for x in range(j, size + 1, j):
                for y in range(j, size + 1, j):
                    
                    neighbourhood = np.bincount(attrs[start_x:x, start_y:y].flatten(), minlength=maximum)
                    neighbourhoods.iloc[index] = list(neighbourhood)[1:]
                    index += 1
                    start_y = y
                start_x = x
                start_y = 0
                
            segregation.append(None)#SpatialDissim(neighbourhoods, cols).statistic)
        columns = ['time'] + [str(k) for k in sizes]
        data = pd.DataFrame(columns=columns, index=[0])
        data.iloc[0] = [self.schedule.time] + segregation
        return data

    def simulate(self, export=False, filename=None):

        while self.schedule.time < self.max_steps:
            self.step()
            if self.convergence_check():
                self.running = False
                break
        
        if export:
            self.save_data()
    
    def save_data(self):

        agents = self.schedule.agents
        headers = ['x', 'y', 'type', 'satisfied', 'homophily', 'fraction_sim']
        data = [0]*len(agents)
        for i, agent in enumerate(agents):
            x, y = agent.pos
            satisfied = agent.satisfied
            agent_type = agent.type
            homophily = agent.homophily
            fraction_sim = agent.fraction_sim
            data[i] = [x, y, agent_type, satisfied, homophily, fraction_sim]

        np.savez(self.filename, headers=headers, data=np.array(data))

    def get_bokeh_vis_data(self):
        
        agents = self.schedule.agents
        data = pd.DataFrame(
            columns=['x', 'y', 'type', 'satisfied', 'homophily', 'fraction_sim'],
            index=range(self.num_agents))
        for i, agent in enumerate(agents):
            x, y = agent.pos
            satisfied = agent.satisfied
            agent_type = agent.type
            homophily = agent.homophily
            fraction_sim = agent.fraction_sim
            data.iloc[i] = [x, y, agent_type, satisfied, homophily, fraction_sim]

        system_data = pd.DataFrame(columns=
            ['time', 'satisfied_fraction', 'avg_fraction_sim', 
            'fraction_sat_1', 'fraction_sat_2'], index=[0])
        system_data.iloc[0] = [self.schedule.time,
            data['satisfied'].mean(), data['fraction_sim'].mean(),
            self.fraction_sat_1, self.fraction_sat_2]
        return data, system_data


if __name__ == "__main__":
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()

    model = Schelling()
    for _ in range(100):
        model.step()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(0.25)