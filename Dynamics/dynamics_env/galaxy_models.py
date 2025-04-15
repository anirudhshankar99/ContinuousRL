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
        if selfpos is not None: selfpos = self.pos
        # assert pos.shape[0] == selfpos.shape[0], f"Dimensions of position vector must be consistent with that of the source potential, position vector shape: {pos.shape}"
        if len(pos.shape) > len(selfpos.shape):
            selfpos = np.expand_dims(selfpos, axis=[i for i in range(len(selfpos.shape), len(pos.shape))])
        elif len(pos.shape) < len(selfpos.shape):
            pos = np.expand_dims(pos, axis=[i for i in range(len(pos.shape), len(selfpos.shape))])
        del_r = selfpos - pos # in pc
        r = np.linalg.norm(del_r, axis=0) # in pc
        a = -G_IN_PC_KMS * self.M / (r**3 + 1e-5) * pos
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
    
    def get_acceleration(self, pos):
        assert len(pos) == len(self.pos), "Dimensions of position vector must be consistent with that of the source potential"
        del_r = self.pos - pos # in pc
        r = np.linalg.norm(del_r) # in pc
        a = -G_IN_PC_KMS * self.M / (r + self.a_b)**2
        theta = np.arccos(del_r[2]/r)
        phi = np.arctan2(del_r[1], del_r[0])
        ax, ay, az = a * np.sin(phi) * np.cos(theta), a * np.sin(phi) * np.sin(theta), a * np.cos(phi)
        return ax, ay, az

model_mapping = {
    'point_source': point_source,
    'bulge': bulge,
}