import gymnasium as gym
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame
from gymnasium import spaces

BALL_START = 50, 50
GOAL_START = 500, 500
FPS = 50
DISP_W, DISP_H = 600, 600
BALL_VELOCITY_CUTOFF = 50
MAX_ATTEMPTS = 30
FORCE_SCALING = [20, 500]
COLLISION_ELASTICITY = 0.9
FRICTION_COEFF = 0.05
STATIC_COLLISION_TYPE = 1
DYNAMIC_COLLISION_TYPE = 2
HOLE_RADIUS = 20
WALL_THICKNESS = 10
STEPS_CUTOFF = 15
MAX_REWARD = 10000

class GolfEnv(gym.Env):
    def __init__(self):
        super(GolfEnv, self).__init__()

        # Pygame setup
        pygame.init()
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((DISP_W, DISP_H))

        # Pymunk physics setup
        self.space = pymunk.Space()

        # Ball setup
        self.ball_radius = 10
        self.ball_mass = 1
        moment = pymunk.moment_for_circle(self.ball_mass, 0, self.ball_radius)
        self.ball_body = pymunk.Body(self.ball_mass, moment)
        self.ball_body.position = BALL_START
        self.ball_shape = pymunk.Circle(self.ball_body, self.ball_radius)
        self.ball_shape.elasticity = COLLISION_ELASTICITY
        self.ball_shape.collision_type = DYNAMIC_COLLISION_TYPE
        self.space.add(self.ball_body, self.ball_shape)
        # self.space.damping = 1 - FRICTION_COEFF

        # Goal (hole)
        self.goal_radius = HOLE_RADIUS
        self.hole_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.hole_body.position = GOAL_START
        self.hole_shape = pymunk.Circle(self.hole_body, self.goal_radius)
        self.hole_shape.collision_type = STATIC_COLLISION_TYPE
        self.space.add(self.hole_body, self.hole_shape)

        # Ground (boundary)
        self.wall_positions = [[(DISP_W/3,0), (DISP_W/3, 2*DISP_H/3)], 
                               [(DISP_W/3*2,DISP_H), (DISP_W/3*2, DISP_H/3)], 
                               [(0,0), (0, DISP_H)], 
                               [(DISP_W,0), (DISP_W, DISP_H)], 
                               [(0,DISP_H), (DISP_W, DISP_H)], 
                               [(0,0), (DISP_W, 0)]]

        self.wall_positions = [[(0,0), (0, DISP_H)], 
                               [(DISP_W,0), (DISP_W, DISP_H)], 
                               [(0,DISP_H), (DISP_W, DISP_H)], 
                               [(0,0), (DISP_W, 0)]]

        self.wall_bodies = []
        self.wall_shapes = []

        for position_pair in self.wall_positions:
            self.wall_bodies.append(pymunk.Body(body_type=pymunk.Body.STATIC))
            self.wall_shapes.append(pymunk.Segment(self.wall_bodies[-1], position_pair[0], position_pair[1], WALL_THICKNESS))
            self.wall_shapes[-1].elasticity = 1
            self.wall_shapes[-1].collision_type = DYNAMIC_COLLISION_TYPE
            self.space.add(self.wall_bodies[-1], self.wall_shapes[-1])

        handler_static = self.space.add_collision_handler(STATIC_COLLISION_TYPE, DYNAMIC_COLLISION_TYPE)
        handler_static.begin = lambda arbiter, space, data: False

        handler_dynamic = self.space.add_collision_handler(DYNAMIC_COLLISION_TYPE, DYNAMIC_COLLISION_TYPE)
        handler_dynamic.begin = lambda arbiter, space, data: True

        # Action space: [angle (0 to 360 degrees), force (0 to 500)]
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)

        # Observation space: Ball position (x, y), distance to hole (x, y)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([DISP_W, DISP_H, DISP_W, DISP_H]),
            dtype=np.float32,
        )

        self.attempts = 0
        self.prev_dist = np.linalg.norm(np.array(self.ball_body.position) - self.hole_body.position.int_tuple)

    def step(self, action, visualize = False):
        """ Apply action: hit the ball with given angle and force """
        angle, force = action[0]*np.pi, self.force_scaling(action[1])
        impulse = (np.cos(angle) * force, np.sin(angle) * force)
        self.ball_body.apply_impulse_at_local_point(impulse)

        # Step physics
        # till it comes to a halt 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        for _ in range(int(np.ceil(np.log(BALL_VELOCITY_CUTOFF/np.linalg.norm(self.ball_body.velocity))/np.log(1-FRICTION_COEFF)))):
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                self.display.fill((255, 255, 255))
                ball_position = self.ball_body.position.int_tuple
                x, y = self.convert_position(ball_position)
                pygame.draw.circle(self.display, (255, 0, 0), (int(x), int(y)), self.ball_radius)

                hole_position = self.hole_body.position.int_tuple
                x, y = self.convert_position(hole_position)
                pygame.draw.circle(self.display, (0, 0, 0), (int(x), int(y)), self.goal_radius)

                for position_pair in self.wall_positions:
                    pygame.draw.line(self.display, (0, 0, 0), self.convert_position(position_pair[0]), self.convert_position(position_pair[1]), WALL_THICKNESS) # 50 + 5/2 as pymunk draws segment width above, and mygame draws around

                pygame.display.update()
                self.clock.tick(FPS)
            self.ball_body.velocity *= (1 - FRICTION_COEFF)
            ball_pos = np.array(self.ball_body.position)
            distance_to_hole = np.linalg.norm(ball_pos - self.hole_body.position.int_tuple)
            if distance_to_hole < self.goal_radius:  # Ball in hole
                reward = (self.prev_dist - distance_to_hole)/MAX_REWARD
                reward += 1/(self.attempts+1)
                obs = np.concatenate([ball_pos, ball_pos-np.array(self.hole_body.position)])
                return obs, reward, True, False, {}
            self.space.step(1/FPS)
        self.ball_body.velocity = (0, 0)

        # Compute new state
        ball_pos = np.array(self.ball_body.position)
        distance_to_hole = np.linalg.norm(ball_pos - self.hole_body.position.int_tuple)

        # Reward function
        reward = (self.prev_dist - distance_to_hole)/MAX_REWARD
        self.prev_dist = distance_to_hole

        # Observation: Ball (x, y), distance to hole
        obs = np.concatenate([ball_pos, ball_pos-np.array(self.hole_body.position)])
        self.attempts += 1
        return obs, reward, False, False, {}

    def reset(self, start=True):
        """ Reset the environment """
        if start: self.ball_body.position = BALL_START
        else: self.ball_body.position = list(np.random.rand(2,)*np.array([DISP_W, DISP_H]))
        self.ball_body.velocity = (0, 0)
        # self.space.step(1 / FPS)  # Advance physics a bit
        # distance_to_hole = np.linalg.norm(np.array(self.ball_body.position) - self.hole_body.position.int_tuple)
        self.attempts = 0
        self.prev_dist = np.linalg.norm(np.array(self.ball_body.position) - self.hole_body.position.int_tuple)
        return np.array([*self.ball_body.position, *(np.array(self.ball_body.position)-np.array(self.hole_body.position)).tolist()])

    def close(self):
        pygame.quit()

    def convert_position(self, position): # pygame origin at topleft, pymunk origin at bottomleft
        return position[0], DISP_H - position[1]
    
    def force_scaling(self, force):
        return (force + 0.5 + FORCE_SCALING[0]/(FORCE_SCALING[1]-FORCE_SCALING[0]))*(FORCE_SCALING[1]-FORCE_SCALING[0])