import gymnasium as gym
from gymnasium import spaces
import pygame
import torch
import numpy as np


class SquashEnv(gym.Env):
    metadata = {"render_modes": ["human",], "render_fps": 4}

    def __init__(self, render_mode:str=None, hyperparameters:dict={}):
        self._init_hyperparameters(hyperparameters)
        if self.size_y_to_x_ratio > 1:
            self.size_x = 1/self.size_y_to_x_ratio
            self.size_y = 1
        else:
            self.size_y = 1/self.size_y_to_x_ratio
            self.size_x = 1
        
        self.max_impulse = self.size_y # covers the length of the field in 1 second
        # self.window_size = 512  # The size of the PyGame window

        # Observations: agent_1 (x,y), agent_2(x,y), ball(x,y), ball_velocity(x,y)
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0., 0., 0., 0., 0., 0., ]),
            high=np.array([self.size_x, self.size_y, self.size_x, self.size_y, self.size_x, self.size_y, self.max_impulse / 0.1]), # eventually remove velocity limit
            dtype=np.float32
        )

        # Actions: move_x, move_y, impulse_to_ball, impulse_angle
        self.action_space = spaces.Box(
            low=np.array([-1., -1., 0., -np.pi / 2,]),
            high=np.array([1., 1., self.max_impulse, np.pi / 2,]), # eventually remove impulse limit
            dtype=np.float32
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent1_location = np.array([self.size_x / 2, self.size_y / 2], dtype=np.float64)
        self._agent2_location = np.array([self.size_x / 2, self.size_y / 2], dtype=np.float64)
        self._ball_location = np.array([self.size_x / 2, self.size_y / 2], dtype=np.float64)
        self._ball_velocity = np.array([0., 0.], dtype=np.float64)
        self._agent2_turn = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._process_actions(action)
        self._agent1_location = np.clip()
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _init_hyperparameters(self, hyperparameters):
        self.size_y_to_x_ratio = 1.5
        self.seed = 0

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        if self.seed != None:
            assert(type(self.seed) == int)

            torch.manual_seed(self.seed)
            print(f"Seed set to {self.seed}")

    def _get_obs(self):
        return np.concat([self._agent1_location,
                          self._agent2_location,
                          self._ball_location,
                          self._ball_velocity,],)

    def _get_info(self):
        return {}
    
    def _process_actions(self, action):
        mask = torch.tensor([1., 1., 0., 0.], dtype=torch.float32)
        antimask = torch.tensor([0., 0., 1., 1.], dtype=torch.float32)
        directions = ((action > 0) * mask).float32()
        rest = action * antimask
        return directions + rest
        
