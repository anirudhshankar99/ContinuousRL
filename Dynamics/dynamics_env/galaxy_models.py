import numpy as np
G_IN_PC_KMS = 4.30091e-3


def add_galaxy_model(model_name, **kwargs):
    assert model_name in model_mapping, "Unsupported galaxy model: %s"%model_name
    return model_mapping[model_name](kwargs)

def point_source(init_params):
    assert 'M' in init_params, "Mass must be supplied to initialize a point source"
    if 'pos' not in init_params:
        return PointSource(init_params['M'])
    return PointSource(init_params['M'], init_params['pos'])

def bulge(init_params):
    assert 'M' in init_params, "Mass must be supplied to initialize a bulge potential"
    assert 'a_b' in init_params, "Bulge scale length must be supplied to initialize a bulge potential"
    if 'pos' not in init_params:
        return PointSource(init_params['M'])
    return PointSource(init_params['M'], init_params['pos'])

def disk(init_params):
    assert 'M' in init_params, "Mass must be supplied to initialize a disk potential"
    assert 'a' in init_params, "Radial scale length must be supplied to initialize a disk potential"
    assert 'b' in init_params, "Vertical scale length must be supplied to initialize a disk potential"
    if 'pos' not in init_params:
        return Disk(init_params['M'], init_params['a'], init_params['b'])
    return Disk(init_params['M'], init_params['a'], init_params['b'], init_params['pos'])

def halo(init_params):
    assert 'v_halo' in init_params, "Asymptotic circular velocity must be supplied to initialize a halo potential"
    assert 'r_c' in init_params, "Core radius must be supplied to initialize a halo potential"
    if 'pos' not in init_params:
        return Halo(init_params['v_halo'], init_params['r_c'])
    return Halo(init_params['v_halo'], init_params['r_c'], init_params['pos'])

class PointSource():
    def __init__(self, M, pos=[0., 0., 0.,]):
        self.M = M # in solar M
        self.pos = np.array(pos) # in pc
    
    def get_field(self, m, pos):
        # In development
        assert pos.shape[0] == len(self.pos), "Dimensions of position vector must be consistent with that of the source potential"
        r = np.linalg.norm((self.pos - pos))
        return -G_IN_PC_KMS * self.M * m / r
    
    def get_acceleration(self, pos, selfpos=None):
        if selfpos is None: selfpos = self.pos
        if len(pos.shape) > len(selfpos.shape):
            selfpos = np.expand_dims(selfpos, axis=[i for i in range(len(selfpos.shape), len(pos.shape))])
        elif len(pos.shape) < len(selfpos.shape):
            pos = np.expand_dims(pos, axis=[i for i in range(len(pos.shape), len(selfpos.shape))])
        del_r = selfpos - pos # in pc
        r = np.linalg.norm(del_r, axis=0) # in pc
        a = G_IN_PC_KMS * self.M / (r**3 + 1e-5) * del_r
        ax, ay, az = np.split(a, 3, axis=0)
        return ax, ay, az
    
class Bulge():
    def __init__(self, M, a_b, pos=[0., 0., 0.,]):
        self.M = M
        self.a_b = a_b
        self.pos = pos

    def get_potential(self, m, pos):
        assert pos.shape[0] == len(self.pos), "Dimensions of position vector must be consistent with that of the source potential"
        r = np.linalg.norm((self.pos - pos))
        return -G_IN_PC_KMS * self.M * m / (r + self.a_b)
    
    def get_acceleration(self, pos, selfpos=None):
        if selfpos is None: selfpos = self.pos
        if len(pos.shape) > len(selfpos.shape):
            selfpos = np.expand_dims(selfpos, axis=[i for i in range(len(selfpos.shape), len(pos.shape))])
        elif len(pos.shape) < len(selfpos.shape):
            pos = np.expand_dims(pos, axis=[i for i in range(len(pos.shape), len(selfpos.shape))])
        del_r = self.pos - pos # in pc
        r = np.linalg.norm(del_r, axis=0) # in pc
        a = G_IN_PC_KMS * self.M / (r * (r + self.a_b)**2) * del_r
        ax, ay, az = np.split(a, 3, axis=0)
        return ax, ay, az

class Halo():
    def __init__(self, v_halo, r_c, pos=[0., 0., 0.,]):
        self.v_halo = v_halo
        self.r_c = r_c
        self.pos = pos

    def get_acceleration(self, pos, selfpos=None):
        if selfpos is None: selfpos = self.pos
        if len(pos.shape) > len(selfpos.shape):
            selfpos = np.expand_dims(selfpos, axis=[i for i in range(len(selfpos.shape), len(pos.shape))])
        elif len(pos.shape) < len(selfpos.shape):
            pos = np.expand_dims(pos, axis=[i for i in range(len(pos.shape), len(selfpos.shape))])
        del_r = self.pos - pos # in pc
        r = np.linalg.norm(del_r, axis=0) # in pc
        r2 = r**2 + self.r_c**2
        a = self.v_halo**2 * del_r / r2
        ax, ay, az = np.split(a, 3, axis=0)
        return ax, ay, az
    
class Disk():
    def __init__(self, M, a, b):
        self.M = M
        self.a = a
        self.b = b
    
    def get_acceleration(self, pos, selfpos=None):
        if selfpos is None: selfpos = self.pos
        if len(pos.shape) > len(selfpos.shape):
            selfpos = np.expand_dims(selfpos, axis=[i for i in range(len(selfpos.shape), len(pos.shape))])
        elif len(pos.shape) < len(selfpos.shape):
            pos = np.expand_dims(pos, axis=[i for i in range(len(pos.shape), len(selfpos.shape))])
        del_r = self.pos - pos # in pc
        # r = np.linalg.norm(del_r, axis=0) # in pc
        R = np.linalg.norm(del_r[:-1], axis=0)
        Z = del_r[-1]
        B = self.b + np.sqrt(Z**2 + self.b**2)
        d = (R**2 + B**2)**1.5
        a = G_IN_PC_KMS * self.M / d * del_r
        ax, ay, az = np.split(a, 3, axis=0)
        return ax, ay, az
    
class Bar():
    def __init__(self, M, a, b, c, omega_p):
        self.M = M
        self.a = a
        self.b = b
        self.c = c
        self.omega_p = omega_p

    def get_acceleration(self, pos, selfpos=None):
        pass

model_mapping = {
    'point_source': point_source,
    'bulge': bulge,
    'halo': halo,
    'disk': disk,
}