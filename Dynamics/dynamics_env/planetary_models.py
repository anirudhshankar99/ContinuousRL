import numpy as np

G_IN_SI = 6.674e-11
YR_TO_SEC = 86400 * 365
SOLAR_MASS = 2e30
EARTH_MASS = 5.97e24
AU_TO_M = 1.496e11

#as of 20th May 2025
PLANET_PHASES = {'earth':29.23, 'mercury':10.37, 'venus':5.32, 'mars':20.98, 'jupiter':0.33, 'saturn':24.37, 'uranus':27.33, 'neptune':29.98}
PLANET_MEAN_ORBIT_RADII = {'earth':1, 'mercury':0.45, 'venus':0.725, 'mars':1.525, 'jupiter':5.205, 'saturn':9.6, 'uranus':19.2, 'neptune':30.2}
PLANET_MASSES = {'mercury':0.0553, 'venus':0.815, 'earth':1, 'mars':0.1075, 'jupiter':317.8, 'saturn':95.2,	'uranus':14.6, 'neptune':17.2}
PLANET_PERIODS = {'mercury':0.240846, 'venus':0.615, 'earth':1, 'mars':1.881, 'jupiter':11.86, 'saturn':29.46, 'uranus':84.01, 'neptune':164.8}
def add_planetary_model(model_name, **kwargs):
    assert model_name in model_mapping, "Unsupported galaxy model: %s"%model_name
    return model_mapping[model_name](kwargs)

def point_source(init_params):
    assert 'M' in init_params, "Mass must be supplied to initialize a point source"
    assert 'phase' in init_params, "Initial phase must be supplied to initialize a point source"
    assert 'orbit_radius' in init_params, "Orbital radius must be supplied to initialize a point source"
    assert 'period' in init_params, "Planetary period must be supplied to initialize a point source"
    return PointSource(init_params['M'], init_params['period'], init_params['phase'], init_params['orbit_radius'])

def planet(init_params):
    assert 'planet_name' in init_params, 'Planets must be supplied with "planet_name" argument'
    assert init_params['planet_name'] in list(PLANET_PHASES.keys()), 'Unknown planet, %s. Available planets: '%init_params['planet_name']+ " ".join(str(x) for x in list(PLANET_PHASES.keys()))
    planet_name = init_params['planet_name']
    return PointSource(PLANET_MASSES[planet_name]*EARTH_MASS, PLANET_PERIODS[planet_name]*YR_TO_SEC, PLANET_PHASES[planet_name], PLANET_MEAN_ORBIT_RADII[planet_name]*AU_TO_M)

def tracer(init_params):
    assert 'phase' in init_params, "Initial phase must be supplied to initialize a point source"
    assert 'orbit_radius' in init_params, "Orbital radius must be supplied to initialize a point source"
    assert 'period' in init_params, "Planetary period must be supplied to initialize a point source"
    return PointSource(0, init_params['period'], init_params['phase'], init_params['orbit_radius'])

class PointSource():
    def __init__(self, M, period, phase, orbit_radius):
        self.M = M # in kg
        self.period = period * YR_TO_SEC # time per revolution
        self.orbit_radius = orbit_radius # in m
        self.sign = 'point_source'
        self.phase = phase # in rad
    
    def get_acceleration(self, pos):
        pos, t = pos[:2], pos[2] # in m, s
        theta = 2 * np.pi * t / self.period + self.phase
        selfpos = np.array([np.cos(theta), np.sin(theta)]) * self.orbit_radius # in m
        del_r = pos - selfpos # in m
        r = np.linalg.norm(del_r, axis=-1) # in m
        a = -G_IN_SI * self.M / (r**3 + 1e-5) * del_r
        ax, ay = np.split(a, 2, axis=-1)
        return ax, ay
    
    def get_position(self, t):
        theta = 2 * np.pi * t / self.period + self.phase
        return np.array([np.cos(theta), np.sin(theta)]) * self.orbit_radius # in m
    
    def get_velocity(self, t):
        if self.orbit_radius == 0:
            return np.zeros((2,))
        speed = np.sqrt(G_IN_SI * SOLAR_MASS / self.orbit_radius)
        theta = 2 * np.pi * t / self.period + self.phase
        position = np.array([np.cos(theta), np.sin(theta)]) * self.orbit_radius # in m
        unit_position_tangent = np.array([-position[...,1] , position[...,0]]) / np.linalg.norm(position)
        return unit_position_tangent * speed
model_mapping = {
    'point_source': point_source,
    'tracer': tracer,
    'planet': planet,
}