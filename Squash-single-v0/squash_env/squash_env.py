import gymnasium as gym
from gymnasium import spaces
import pygame
import torch
import numpy as np
from matplotlib.colors import hsv_to_rgb

class SquashEnv(gym.Env):

    def __init__(self, hyperparameters:dict={}):
        self._init_hyperparameters(hyperparameters)
        self.render_modes = ['human', 'logs', 'none']
        if self.render_mode.lower() not in self.render_modes: 
            print('The render_mode must be either \'human\', \'logs\' or \'none\'. Defaulting to none.')
            self.render_mode = 'none'
        if self.size_y_to_x_ratio > 1:
            self.size_x = 1/self.size_y_to_x_ratio
            self.size_y = 1
        else:
            self.size_y = 1/self.size_y_to_x_ratio
            self.size_x = 1
        self.agent_speed = 0.4
        self._ball_size = self.size_x / 30 # 30 balls can fill the width of the field
        self._agent_size = self._ball_size # agent takes the same size
        self._wall_width = self._ball_size / 5 # 20% of the ball's size
        self.max_impulse = self.size_y # covers the length of the field in 1 second
        self.min_impulse = 0.25 * self.size_y # covers the length of the field in 4 seconds
        self._window_size_scaling = 800
        self._agent_colours = (np.stack(self._get_agent_colours(), axis=0)*255).astype(int)

        # Observations: agent position(x,y), ball(x,y), distance_to_ball, ball_velocity(v_r,v_theta)
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0., 0., 0., 0., -np.pi / 2]),
            high=np.array([self.size_x, self.size_y, self.size_x, self.size_y, np.sqrt(self.size_x**2 + self.size_y**2), self.max_impulse / 0.1, np.pi / 2]), # eventually remove velocity limit
            dtype=np.float32
        )

        # Actions: agent move (x,y), impulse_to_ball, impulse_angle
        self.action_space = spaces.Box(
            low=np.array([-1., -1., self.min_impulse, -np.pi / 2,]),
            high=np.array([1., 1., self.max_impulse, np.pi / 2,]), # eventually remove impulse limit
            dtype=np.float32
        )

        self._wall_positions = [[np.array([0,0]), np.array([0, self.size_y])], 
                               [np.array([self.size_x,0]), np.array([self.size_x, self.size_y])], 
                               [np.array([0,self.size_y]), np.array([self.size_x, self.size_y])], 
                               [np.array([0,0]), np.array([self.size_x, 0])]]

        self.display = None
        self.clock = None
        if self.render_mode == 'logs':
            self._logs = []

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._agent_location = np.array([self.size_x / 2, self.size_y / 2], dtype=np.float64)
        if self.ball_start == 'random' or self.ball_start == 'r':
            ball_loc_r_theta = np.ones_like(self._ball_location) * np.array([np.min([self.size_x, self.size_y]).item()*0.5, (np.random.rand(1,) * np.pi * 2).item()])
            self._ball_location = np.array([ball_loc_r_theta[0]*np.cos(ball_loc_r_theta[1]).item(), ball_loc_r_theta[0]*np.sin(ball_loc_r_theta[1]).item()]) + self._agent_location
        elif self.ball_start =='center' or self.ball_start == 'c':
            self._ball_location = np.array([self.size_x / 2, self.size_y / 2], dtype=np.float64)
        self._ball_velocity = np.array([0.,0.])

        observation = self._get_obs()
        # info = self._get_info() # to include number of volleys

        if self.render_mode == 'human':
            self.render()
        if self.render_mode == 'logs':
            self._log_entry('[RESET] ')

        self._prev_dist_to_ball = np.linalg.norm(self._agent_location - self._ball_location)

        return observation, {}

    def step(self, action):
        """
        Action is of the form { 'agent_x': (action_dim, ), ...}
        """
        reward = 0
        processed_actions = self._process_actions(action)

        self._ball_velocity[1] = self._ball_rebound()

        self._agent_location = np.clip(
            self._agent_location + processed_actions[:-2],
            [self._ball_size, self._ball_size],
            [self.size_x - self._ball_size, self.size_y - self._ball_size]
            )

        # ball impact    
        if np.linalg.norm(self._ball_location - self._agent_location)<self._ball_size:
            self._ball_velocity = self._2d_collision_elastic(self._ball_velocity, action[-2:])
            # self._ball_velocity = self._impart_impulse(action[-2:])
            reward += 1

        # ball movement
        self._ball_location = self._ball_location + np.stack([self._ball_velocity[0]*np.cos(self._ball_velocity[1]), self._ball_velocity[0]*np.sin(self._ball_velocity[1])]) / self.fps

        observation = self._get_obs()
        # info = self._get_info()

        # ball_distance = np.linalg.norm(self._agent_location - self._ball_location)
        # reward = (self._prev_dist_to_ball - ball_distance)

        if self.render_mode == "human":
            self.render()
        
        done=self._ball_out_of_bounds()
        # if np.linalg.norm(self._ball_location - self._agent_location)<self._ball_size: done=True

        return observation, reward, done, False, {}

    def render(self):
        if self.display is None:
            pygame.init()
            pygame.display.init()
            self.font = pygame.font.Font('freesansbold.ttf', 32)
            self.display = pygame.display.set_mode((self._window_size_scaling*self.size_x, self._window_size_scaling*self.size_y))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.display.fill((255, 255, 255))

        pygame.draw.circle(self.display, (255, 0, 0), (int(self._ball_location[0]*self._window_size_scaling), int(self._ball_location[1]*self._window_size_scaling)), self._ball_size*self._window_size_scaling)

        pygame.draw.circle(self.display, self._agent_colours[0], (int(self._agent_location[0]*self._window_size_scaling), int(self._agent_location[1]*self._window_size_scaling)), self._agent_size*self._window_size_scaling)

        for position_pair in self._wall_positions:
            pygame.draw.line(self.display, (0, 0, 0), position_pair[0].astype(int).tolist(), position_pair[1].astype(int).tolist(), int(self._wall_width * self._window_size_scaling))
        pygame.display.update()
        self.clock.tick(self.fps)

    def close(self):
        if self.display is not None:
            pygame.display.quit()
            pygame.quit()

    def _init_hyperparameters(self, hyperparameters):
        self.size_y_to_x_ratio = 1.5
        self.seed = 0
        self.num_agents = 1
        self.fps = 30
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.render_mode = 'none'
        self.ball_start = 'center'

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + '%s'%val)

        if self.seed != None:
            assert(type(self.seed) == int)

            torch.manual_seed(self.seed)
            print(f"Seed set to {self.seed}")

    def _get_obs(self):
        obs = np.array(self._agent_location.tolist()+self._ball_location.tolist()+[np.linalg.norm(self._ball_location - self._agent_location)]+self._ball_velocity.tolist())
        return obs

    def _get_info(self):
        return {'episode':{'r':self.reward, 'l':0}}
    
    def _process_actions(self, action):
        clipped_action =  np.clip(action, -1, 1)
        clipped_action[-2:] += 1
        clipped_action*= np.array([self.agent_speed / self.fps for _ in range(len(action[:-2]))]+[self.max_impulse / 2, np.pi])
        return clipped_action
        return (action / np.concat([np.abs(action[:-2]), action[-2:]]) * self.agent_speed / self.fps)
    
    def _euclidian_dist(self, position_1, position_2):
        return np.linalg.norm(position_1 - position_2)
    
    def _ball_rebound(self):
        def _reflect(theta, phi): return (2 * phi - theta) % (2 * np.pi)
        left, right, up, down = self._ball_location[0] < self._ball_size, self._ball_location[0] > self.size_x - self._ball_size, self._ball_location[1] < self._ball_size, self._ball_location[1] > self.size_y - self._ball_size
        if up:
            if left: 
                if self.render_mode == 'logs': self._log_entry('[REBOUND] off the top-left. ')
                return _reflect(_reflect(self._ball_velocity[1], 3 * np.pi / 2), 0)
            elif right:
                if self.render_mode == 'logs': self._log_entry('[REBOUND] off the top-right. ')
                return _reflect(_reflect(self._ball_velocity[1], np.pi / 2), 0)
            else: 
                if self.render_mode == 'logs': self._log_entry('[REBOUND] off the top. ')
                return _reflect(self._ball_velocity[1], np.pi)
        # elif down:
        #     if left: 
        #         if self.render_mode == 'logs': self._log_entry('[REBOUND] off the bottom-left. ')
        #         return _reflect(_reflect(self._ball_velocity[1], 3 * np.pi / 2), np.pi)
        #     elif right:
        #         if self.render_mode == 'logs': self._log_entry('[REBOUND] off the bottom-right. ')
        #         return _reflect(_reflect(self._ball_velocity[1], np.pi / 2), np.pi)
        #     else: 
        #         if self.render_mode == 'logs': self._log_entry('[REBOUND] off the bottom. ')
        #         return _reflect(self._ball_velocity[1], np.pi)
        elif left: 
            if self.render_mode == 'logs': self._log_entry('[REBOUND] off the left wall. ')
            return _reflect(self._ball_velocity[1], 3 * np.pi / 2)
        elif right: 
            if self.render_mode == 'logs': self._log_entry('[REBOUND] off the right wall. ')
            return _reflect(self._ball_velocity[1], np.pi / 2)
        return self._ball_velocity[1]
    
    def _get_agent_colours(self):
        colors = []
        for i in np.arange(0., 360., 360. / self.num_agents):
            h = i / 360.
            l = (50 + np.random.rand() * 10) / 100.
            s = (90 + np.random.rand() * 10) / 100.
            colors.append(hsv_to_rgb([h, l, s]))
        return colors
    
    def _log_entry(self, state_description):
        _log = state_description
        _log += 'agent is at ('
        for coordinate in self._agent_location:
            _log += '%.2f,'%coordinate
        _log += '), '
        _log += 'the ball is at ('
        for coordinate in self._ball_location:
            _log += '%.2f,'%coordinate
        _log += ').'
        print(_log)
        self._logs.append(_log)

    def _impart_impulse(self, impulse): return impulse

    def _2d_collision_elastic(self, initial_velocity, impulse):
        ball_velocity_x, ball_velocity_y = initial_velocity[0] * np.cos(initial_velocity[1]), initial_velocity[0] * np.sin(initial_velocity[1])
        impulse_x, impulse_y = impulse[0] * np.cos(impulse[1]), impulse[0] * np.sin(impulse[1])
        final_ball_velocity_x, final_ball_velocity_y = ball_velocity_x + impulse_x, ball_velocity_y + impulse_y
        return np.stack([np.sqrt(final_ball_velocity_x ** 2 + final_ball_velocity_y ** 2), np.arctan2(final_ball_velocity_y, final_ball_velocity_x)])
    
    def _ball_out_of_bounds(self):
        return self._ball_location[1] > self.size_y - self._ball_size