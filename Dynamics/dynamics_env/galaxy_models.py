import numpy as np

G_IN_PC_KMS = 4.30091e-3
MYR_TO_SEC = 86400 * 365 * 1e6

def add_galaxy_model(model_name, **kwargs):
    assert model_name in model_mapping, "Unsupported galaxy model: %s"%model_name
    return model_mapping[model_name](kwargs)

def point_source(init_params):
    assert 'M' in init_params, "Mass must be supplied to initialize a point source"
    if 'pos' not in init_params:
        return PointSource(init_params['M'])
    return PointSource(init_params['M'], init_params['pos'])

def tracer(init_params):
    if 'pos' not in init_params:
        return PointSource(M=0)
    return PointSource(M=0, pos=init_params['pos'])

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

def bar(init_params):
    assert 'M' in init_params, "Mass must be supplied to initialize a bar potential"
    assert 'a' in init_params, "Semi-axis length a must be supplied to initialize a disk potential"
    assert 'b' in init_params, "Semi-axis length b must be supplied to initialize a disk potential"
    assert 'c' in init_params, "Semi-axis length c must be supplied to initialize a disk potential"
    assert 'omega_p' in init_params, "Pattern speed must be supplied to initialize a disk potential"
    if 'pos' not in init_params:
        return Bar(init_params['M'], init_params['a'], init_params['b'], init_params['c'], init_params['omega_p'])
    return Bar(init_params['M'], init_params['a'], init_params['b'], init_params['c'], init_params['omega_p'], init_params['pos'])

class PointSource():
    def __init__(self, M, pos=[0., 0., 0.,]):
        self.M = M # in solar M
        self.pos = np.array(pos) # in pc
        self.sign = 'point_source'
    
    def get_field(self, m, pos):
        # In development
        assert pos.shape[0] == len(self.pos), "Dimensions of position vector must be consistent with that of the source potential"
        r = np.linalg.norm((self.pos - pos), axis=-1)
        return -G_IN_PC_KMS * self.M * m / r
    
    def get_acceleration(self, pos, selfpos=None):
        if selfpos is None: selfpos = self.pos
        if len(pos.shape) > len(selfpos.shape):
            selfpos = np.expand_dims(selfpos, axis=[i for i in range(len(selfpos.shape), len(pos.shape))])
        elif len(pos.shape) < len(selfpos.shape):
            pos = np.expand_dims(pos, axis=[i for i in range(len(pos.shape), len(selfpos.shape))])
        del_r = pos - selfpos # in pc
        # if np.linalg.norm(del_r, axis=-1) < origin_capture_delta:
        #     return np.split(np.ones_like(del_r) * np.inf, 3, axis=-1)
        r = np.linalg.norm(del_r, axis=-1) # in pc
        a = -G_IN_PC_KMS * self.M / (r**3 + 1e-5) * del_r
        ax, ay, az = np.split(a, 3, axis=-1)
        # if np.linalg.norm(a, axis=-1) > origin_capture_acc:
        #     return np.split(np.ones_like(a) * np.inf, 3, axis=-1)
        return ax, ay, az
    
class Bulge():
    def __init__(self, M, a_b, pos=[0., 0., 0.,]):
        self.M = M
        self.a_b = a_b
        self.pos = np.array(pos)
        self.sign = 'bulge'

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
        del_r = selfpos - pos # in pc
        r = np.linalg.norm(del_r, axis=0) # in pc
        a = G_IN_PC_KMS * self.M / (r * (r + self.a_b)**2) * del_r
        ax, ay, az = np.split(a, 3, axis=0)
        return ax, ay, az

class Halo():
    def __init__(self, v_halo, r_c, pos=[0., 0., 0.,]):
        self.v_halo = v_halo
        self.r_c = r_c
        self.pos = np.array(pos)
        self.sign = 'halo'

    def get_acceleration(self, pos, selfpos=None):
        if selfpos is None: selfpos = self.pos
        if len(pos.shape) > len(selfpos.shape):
            selfpos = np.expand_dims(selfpos, axis=[i for i in range(len(selfpos.shape), len(pos.shape))])
        elif len(pos.shape) < len(selfpos.shape):
            pos = np.expand_dims(pos, axis=[i for i in range(len(pos.shape), len(selfpos.shape))])
        del_r = selfpos - pos # in pc
        r = np.linalg.norm(del_r, axis=0) # in pc
        r2 = r**2 + self.r_c**2
        a = self.v_halo**2 * del_r / r2
        ax, ay, az = np.split(a, 3, axis=0)
        return ax, ay, az
    
class Disk():
    def __init__(self, M, a, b, pos=[0., 0., 0.]):
        self.M = M
        self.a = a
        self.b = b
        self.pos = np.array(pos)
        self.sign = 'disk'
    
    def get_acceleration(self, pos, selfpos=None):
        if selfpos is None: selfpos = self.pos
        if len(pos.shape) > len(selfpos.shape):
            selfpos = np.expand_dims(selfpos, axis=[i for i in range(len(selfpos.shape), len(pos.shape))])
        elif len(pos.shape) < len(selfpos.shape):
            pos = np.expand_dims(pos, axis=[i for i in range(len(pos.shape), len(selfpos.shape))])
        del_r = selfpos - pos # in pc
        # r = np.linalg.norm(del_r, axis=0) # in pc
        R = np.linalg.norm(del_r[:-1], axis=0)
        Z = del_r[-1]
        B = self.b + np.sqrt(Z**2 + self.b**2)
        d = (R**2 + B**2)**1.5
        a = G_IN_PC_KMS * self.M / d * del_r
        ax, ay, az = np.split(a, 3, axis=0)
        return ax, ay, az
    
class Bar():
    def __init__(self, M, a, b, c, omega_p, pos=[0., 0., 0.,]):
        self.M = M
        self.a = a
        self.b = b
        self.c = c
        self.omega_p = omega_p / MYR_TO_SEC
        self.pos = np.array(pos)
        self.sign = 'bar'

    def get_acceleration(self, pos, selfpos=None):
        assert pos.shape[-1] == 4, 'Calculation of bar potential requires also supplying temporal position'
        spatial_pos, t = pos[...,:3], pos[...,3:]
        if selfpos is None: selfpos = self.pos

        del_r = spatial_pos - selfpos
        theta = self.omega_p * t
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array(np.stack([np.concat([cos_t, -sin_t, np.zeros_like(theta)], axis=-1),
                                    np.concat([sin_t, cos_t, np.zeros_like(theta)], axis=-1),
                                    np.concat([np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta)], axis=-1)], axis=-2))
        rotated_del_r = np.einsum('...j,ij->...i', del_r, rotation_matrix)
        semi_axes = np.array([self.a, self.b, self.c])
        xi_eta_zeta = rotated_del_r / semi_axes
        R_eff = np.linalg.norm(xi_eta_zeta, axis=-1)
        A = -G_IN_PC_KMS * self.M * xi_eta_zeta / (semi_axes**2 * (R_eff**2 + 1e-5)**(1.5))
        rotation_matrix_T = np.transpose(rotation_matrix, axes=(-2,-1))
        A_intertial_frame = np.einsum('...j,ij->...i', A, rotation_matrix_T)
        ax, ay, az = np.split(A_intertial_frame, 3, axis=-1)
        return ax, ay, az

model_mapping = {
    'point_source': point_source,
    'bulge': bulge,
    'halo': halo,
    'disk': disk,
    'tracer': tracer,
    'bar':bar,
}