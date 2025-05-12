import numpy as np

G_IN_SI = 6.674e-11
YR_TO_SEC = 86400 * 365

def add_planetary_model(model_name, **kwargs):
    assert model_name in model_mapping, "Unsupported galaxy model: %s"%model_name
    return model_mapping[model_name](kwargs)

def point_source(init_params):
    assert 'M' in init_params, "Mass must be supplied to initialize a point source"
    assert 'phase' in init_params, "Initial phase must be supplied to initialize a point source"
    assert 'orbit_radius' in init_params, "Orbital radius must be supplied to initialize a point source"
    assert 'period' in init_params, "Planetary period must be supplied to initialize a point source"
    return PointSource(init_params['M'], init_params['period'], init_params['phase'], init_params['orbit_radius'])

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
    
    def get_velocity(self, t, speed):
        theta = 2 * np.pi * t / self.period + self.phase
        position = np.array([np.cos(theta), np.sin(theta)]) * self.orbit_radius # in m
        unit_position_tangent = np.array([-position[...,1] , position[...,0]]) / np.linalg.norm(position)
        return unit_position_tangent * speed
model_mapping = {
    'point_source': point_source,
    'tracer': tracer,
}