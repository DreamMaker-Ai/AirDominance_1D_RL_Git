"""
This code may be called from test_env.py
Gym environment for 1D reinforcement learning problem inspored by RAND corporations report:
    "AirDominance Through Machine Learning, A Preliminary Exploration of
    Artificial Intelligence-Assisted Mission Planning", RAND Corporation, 2020
        https://www.rand.org/pubs/research_reports/RR4311.html
"""
import gym
import numpy as np
import pygame
import math
import matplotlib.pyplot as plt

"""
Define breakdown of mission conditions
"""
MISSION_CONDITIONS = ['w1', 'w2', 'w3', 'l1', 'l2']
MISSION_RATIO = [1, 1, 1, 1, 1]


class Fighter:
    FIGHTER_MIN_FIRING_RANGE = 10.  # km
    FIGHTER_MAX_FIRING_RANGE = 40.  # km
    FIGHTER_ALLOWABLE_NEGATIVE_INGRESS = -10  # km

    def __init__(self):
        # Specifications
        self.alive = 1
        self.speed = 740.  # km/h
        self.ingress = None  # km
        self.previous_ingress = None  # km
        self.min_firing_range = self.FIGHTER_MIN_FIRING_RANGE
        self.max_firing_range = self.FIGHTER_MAX_FIRING_RANGE
        self.firing_range = None
        self.sskp = 1.0  # Assume perfect, but explicitly not used in the program
        self.allowable_negative_ingress = self.FIGHTER_ALLOWABLE_NEGATIVE_INGRESS


class Jammer:
    JAMMER_ALLOWABLE_NEGATIVE_INGRESS = -10  # km

    def __init__(self):
        # Specifications
        self.alive = 1
        self.jam_range = 30.  # km
        self.speed = 740.  # km/h
        self.ingress = None  # km
        self.previous_ingress = None  # km
        self.on = None
        self.previous_on = None
        self.jam_effectiveness = 0.7  # Reduce ratio to the adversarial SAM range
        self.allowable_negative_ingress = self.JAMMER_ALLOWABLE_NEGATIVE_INGRESS


class SAM:
    SAM_MIN_FIRING_RANGE = 10.  # km
    SAM_MAX_FIRING_RANGE = 40.  # km
    SAM_MAX_OFFSET = 100.  # km

    def __init__(self):
        self.alive = 1
        self.min_firing_range = self.SAM_MIN_FIRING_RANGE
        self.max_firing_range = self.SAM_MAX_FIRING_RANGE
        self.firing_range = None
        self.jammed_firing_range = None
        self.offset = None
        self.max_offset = self.SAM_MAX_OFFSET  # km
        self.sskp = 1.0  # Assume perfect, but explicitly not used in the program


class MyEnv(gym.Env):
    # For simulation
    SPACE_X = 100.  # km, positive size of battle space
    SPACE_Y = SPACE_X  # km
    SPACE_OFFSET = -10.  # km, negative size of battle space

    ACTION_DIM = 4
    ACTION_LIST = [[0, 0], [0, 1], [1, 0], [1, 1]]

    """
    ACTION_DIM = 9
    ACTION_LIST = [[-1, -1], [-1, 0], [-1, 1],
                   [0, -1], [0, 0], [0, 1],
                   [1, -1], [1, 0], [1, 1]]
    """

    # For rendering
    WIDTH = 800
    HEIGHT = 800

    RED = (255, 0, 0, 255)
    GREEN = (0, 255, 0, 255)
    BLUE = (0, 0, 255, 255)
    RADIUS = 10

    RED_ZONE = (255, 0, 0, 64)
    JAMMED_RED_ZONE = (255, 0, 0, 96)
    GREEN_ZONE = (0, 255, 0, 64)
    BLUE_ZONE = (0, 0, 255, 64)

    # For screen shot
    SHOT_SHAPE = (80, 80)

    def __init__(self):
        super(MyEnv, self).__init__()
        self.width = self.WIDTH
        self.height = self.HEIGHT

        self.shot_shape = self.SHOT_SHAPE

        self.space_x = self.SPACE_X - self.SPACE_OFFSET
        self.space_y = self.SPACE_Y - self.SPACE_OFFSET
        self.space_offset = self.SPACE_OFFSET

        self.to_screen_x = self.width / self.space_x  # Transform battle space axis to render axis
        self.to_screen_y = self.height / self.space_y
        self.to_screen_offset = self.space_offset * self.to_screen_x

        self.fighter = Fighter()
        self.fighter.color = self.BLUE
        self.fighter.radius = self.RADIUS
        self.fighter.zone_color = self.BLUE_ZONE

        self.jammer = Jammer()
        self.jammer.color = self.GREEN
        self.jammer.radius = self.RADIUS
        self.jammer.zone_color = self.GREEN_ZONE

        self.sam = SAM()
        self.sam.color = self.RED
        self.sam.radius = self.RADIUS
        self.sam.zone_color = self.RED_ZONE
        self.sam.jammed_zone_color = self.JAMMED_RED_ZONE

        self.mission_conditions = MISSION_CONDITIONS
        mission_ratio = np.array(MISSION_RATIO)
        self.mission_probability = mission_ratio / np.sum(mission_ratio)

        self.dt = 1 / self.fighter.speed  # simulation step
        self.max_steps = int(self.sam.max_offset // (self.dt * self.fighter.speed) * 3)  # 300
        self.action_interval = 1
        self.resolution = self.dt * self.fighter.speed * self.action_interval  # resolution of simulations
        self.sam.firing_range_lower_bound = math.ceil((self.resolution * 2) / 0.3)
        if self.sam.firing_range_lower_bound > self.jammer.jam_range:
            raise Exception('Error! Resolution is too big! Reduce action interval!')

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

        # Define discrete action space
        self.action_space = gym.spaces.Discrete(self.ACTION_DIM)

        # Define continuous observation space
        low = np.array([0.] * 8, dtype=np.float32)
        low[0] = (self.fighter.allowable_negative_ingress - self.resolution) / self.sam.max_offset
        low[2] = (self.jammer.allowable_negative_ingress - self.resolution) / self.sam.max_offset
        high = np.array([1.] * 8, dtype=np.float32)
        high[0] = 1 + self.resolution / self.sam.max_offset
        high[2] = 1 + self.resolution / self.sam.max_offset
        self.observation_space = gym.spaces.Box(low=low, high=high)

        self.reset()

    def reset_fighter(self):
        # Battle space coordinate
        self.fighter.alive = 1
        self.fighter.ingress = 0
        low = self.fighter.allowable_negative_ingress + self.resolution
        self.fighter.ingress = np.float(np.random.randint(low=low, high=0))
        # self.fighter.previous_ingress = 0.0
        # self.fighter.firing_range = \
        #    self.fighter.min_firing_range + \
        #    np.random.random() * (self.fighter.max_firing_range - self.fighter.min_firing_range)  # km

    def reset_render_fighter(self):
        # Render coordinate
        self.fighter.screen_x = self.fighter.ingress * self.to_screen_x - self.to_screen_offset
        self.fighter.screen_y = self.space_y / 2.1 * self.to_screen_x
        self.fighter.screen_firing_range = self.fighter.firing_range * self.to_screen_x

        # Fighter circle
        width = self.fighter.radius * 2
        height = self.fighter.radius * 2
        self.fighter.surface = pygame.Surface((width, height))
        pygame.draw.circle(surface=self.fighter.surface,
                           center=(self.fighter.radius, self.fighter.radius),
                           color=self.fighter.color, radius=self.fighter.radius)

        # Range circle
        width = self.fighter.screen_firing_range * 2
        height = self.fighter.screen_firing_range * 2
        self.fighter.surface_range = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(surface=self.fighter.surface_range,
                           center=(self.fighter.screen_firing_range, self.fighter.screen_firing_range),
                           color=self.fighter.zone_color, radius=self.fighter.screen_firing_range)

    def reset_jammer(self):
        # Battle space coordinate
        self.jammer.alive = 1
        self.jammer.ingress = 0.0
        low = self.jammer.allowable_negative_ingress + self.resolution
        self.jammer.ingress = np.float(np.random.randint(low=low, high=0))
        # self.jammer.previous_ingress = 0.0
        self.jammer.on = 0
        self.jammer.previous_on = 0

    def reset_render_jammer(self):
        # Render coordinate
        self.jammer.screen_x = self.jammer.ingress * self.to_screen_x - self.to_screen_offset
        self.jammer.screen_y = self.space_y / 1.9 * self.to_screen_x
        self.jammer.screen_jam_range = self.jammer.jam_range * self.to_screen_x

        # Jammer circle
        width = self.jammer.radius * 2
        height = self.jammer.radius * 2
        self.jammer.surface = pygame.Surface((width, height))
        pygame.draw.circle(surface=self.jammer.surface,
                           center=(self.jammer.radius, self.jammer.radius),
                           color=self.jammer.color, radius=self.jammer.radius)

        # Range circle
        width = self.jammer.screen_jam_range * 2
        height = self.jammer.screen_jam_range * 2
        self.jammer.surface_range = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(surface=self.jammer.surface_range,
                           center=(self.jammer.screen_jam_range, self.jammer.screen_jam_range),
                           color=self.jammer.zone_color, radius=self.jammer.screen_jam_range)

    def reset_sam(self):
        # Battle space coordinate
        self.sam.alive = 1
        # self.sam.firing_range = \
        #    self.sam.min_firing_range + \
        #    np.random.random() * (self.sam.max_firing_range - self.sam.min_firing_range)  # km
        # self.sam.jammed_firing_range = self.sam.firing_range * 0.7  # km
        # self.sam.min_offset = \
        #    max(self.sam.firing_range, self.fighter.firing_range, self.jammer.jam_range) + 5.0  # km
        # self.sam.offset = \
        #    self.sam.min_offset + np.random.random() * (self.sam.max_offset - self.sam.min_offset)  # km

    def reset_render_sam(self):
        # Render coordinate
        self.sam.screen_x = self.sam.offset * self.to_screen_x - self.to_screen_offset
        self.sam.screen_y = self.space_y / 2.0 * self.to_screen_x
        self.sam.screen_firing_range = self.sam.firing_range * self.to_screen_x
        self.sam.screen_jammed_firing_range = self.sam.jammed_firing_range * self.to_screen_x

        # SAM circle
        width = self.sam.radius * 2
        height = self.sam.radius * 2
        self.sam.surface = pygame.Surface((width, height))
        pygame.draw.circle(surface=self.sam.surface,
                           center=(self.sam.radius, self.sam.radius),
                           color=self.sam.color, radius=self.sam.radius)

        # Range circle
        width = self.sam.screen_firing_range * 2
        height = self.sam.screen_firing_range * 2
        self.sam.surface_range = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(surface=self.sam.surface_range,
                           center=(self.sam.screen_firing_range, self.sam.screen_firing_range),
                           color=self.sam.zone_color, radius=self.sam.screen_firing_range)

        # Jammed range circle
        width = self.sam.screen_jammed_firing_range * 2
        height = self.sam.screen_jammed_firing_range * 2
        self.sam.surface_jammed_range = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(surface=self.sam.surface_jammed_range,
                           center=(self.sam.screen_jammed_firing_range, self.sam.screen_jammed_firing_range),
                           color=self.sam.jammed_zone_color, radius=self.sam.screen_jammed_firing_range)

    def condition_w1(self):
        sam_firing_range_list = np.arange(self.jammer.jam_range + self.resolution,
                                          self.sam.max_firing_range + self.resolution,
                                          self.resolution)
        if sam_firing_range_list[-1] > self.sam.max_firing_range:
            sam_firing_range_list = sam_firing_range_list[:-1]

        self.sam.firing_range = np.random.choice(sam_firing_range_list[:-1])
        self.sam.jammed_firing_range = self.sam.firing_range * self.jammer.jam_effectiveness

        fighter_firing_range_index = np.where(sam_firing_range_list > self.sam.firing_range)[0]
        fighter_firing_range_list = sam_firing_range_list[fighter_firing_range_index]
        self.fighter.firing_range = np.random.choice(fighter_firing_range_list)

        """
        self.sam.firing_range = \
            self.jammer.jam_range + \
            (self.sam.max_firing_range - self.jammer.jam_range) * np.random.random()

        self.sam.jammed_firing_range = self.sam.firing_range * self.jammer.jam_effectiveness

        self.fighter.firing_range = \
            self.sam.firing_range + \
            (self.fighter.max_firing_range - self.sam.firing_range) * np.random.random()
        """

        sam_min_offset = \
            max(self.sam.firing_range, self.fighter.firing_range, self.jammer.jam_range) + \
            self.resolution * 2

        self.sam.offset = sam_min_offset + np.random.random() * (self.sam.max_offset - sam_min_offset)
        self.sam.offset = np.round(self.sam.offset)

        condition = (self.sam.max_firing_range >= self.sam.firing_range > self.jammer.jam_range) and \
                    (self.fighter.max_firing_range >= self.fighter.firing_range > self.sam.firing_range) and \
                    (self.fighter.firing_range >= self.sam.firing_range + self.resolution)

        if not condition:
            raise Exception('Error in condition_w1_generator')

    def condition_w2(self):
        sam_firing_range_list = np.arange(self.sam.firing_range_lower_bound,
                                          self.jammer.jam_range,
                                          self.resolution)
        self.sam.firing_range = np.random.choice(sam_firing_range_list[1:])
        self.sam.jammed_firing_range = self.sam.firing_range * self.jammer.jam_effectiveness

        cond = (sam_firing_range_list < self.sam.firing_range) & \
               (sam_firing_range_list > self.sam.jammed_firing_range + self.resolution)
        fighter_firing_range_list_index = np.where(cond)[0]
        fighter_firing_range_index = np.random.choice(fighter_firing_range_list_index)
        self.fighter.firing_range = sam_firing_range_list[fighter_firing_range_index]

        """
        self.sam.firing_range = \
            self.sam.firing_range_lower_bound + \
            (self.jammer.jam_range - self.sam.firing_range_lower_bound) * np.random.random()

        self.sam.jammed_firing_range = self.sam.firing_range * self.jammer.jam_effectiveness

        self.fighter.firing_range = self.sam.jammed_firing_range + (
                self.sam.firing_range - self.sam.jammed_firing_range) * np.random.random()
        """

        sam_min_offset = \
            max(self.sam.firing_range, self.fighter.firing_range, self.jammer.jam_range) \
            + self.resolution * 2

        self.sam.offset = sam_min_offset + np.random.random() * (self.sam.max_offset - sam_min_offset)
        self.sam.offset = np.round(self.sam.offset)

        condition = (self.jammer.jam_range > self.sam.firing_range >
                     self.fighter.firing_range > self.sam.jammed_firing_range) and \
                    (self.fighter.firing_range >= self.sam.jammed_firing_range + self.resolution)

        if not condition:
            raise Exception('Error in condition_w2_generator')

    def condition_w3(self):
        sam_firing_range_list = np.arange(self.sam.firing_range_lower_bound,
                                          self.jammer.jam_range,
                                          self.resolution)
        self.sam.firing_range = np.random.choice(sam_firing_range_list[:-1])
        self.sam.jammed_firing_range = self.sam.firing_range * self.jammer.jam_effectiveness

        fighter_firing_range_index = np.where(sam_firing_range_list > self.sam.firing_range)[0]
        fighter_firing_range_list = sam_firing_range_list[fighter_firing_range_index]
        self.fighter.firing_range = np.random.choice(fighter_firing_range_list)

        """
        self.sam.firing_range = \
            self.sam.firing_range_lower_bound + \
            (self.jammer.jam_range - self.sam.firing_range_lower_bound) * np.random.random()

        self.sam.jammed_firing_range = self.sam.firing_range * self.jammer.jam_effectiveness

        self.fighter.firing_range = \
            self.sam.firing_range + \
            (self.sam.max_firing_range - self.sam.firing_range) * np.random.random()
        """

        sam_min_offset = \
            max(self.sam.firing_range, self.fighter.firing_range, self.jammer.jam_range) \
            + self.resolution * 2

        self.sam.offset = \
            sam_min_offset + np.random.random() * (self.sam.max_offset - sam_min_offset)
        self.sam.offset = np.round(self.sam.offset)

        condition = (self.jammer.jam_range > self.sam.firing_range) and \
                    (self.fighter.max_firing_range > self.fighter.firing_range > self.sam.firing_range) and \
                    (self.fighter.firing_range >= self.sam.firing_range + self.resolution)

        if not condition:
            raise Exception('Error in condition_w3_generator')

    def condition_l1(self):
        fighter_firing_range_list = np.arange(0, self.fighter.max_firing_range, self.resolution)
        self.fighter.firing_range = np.random.choice(fighter_firing_range_list)

        sam_firing_range_list = np.arange(
            np.max([self.fighter.firing_range, self.jammer.jam_range]) + self.resolution,
            self.sam.max_firing_range + self.resolution,
            self.resolution)
        self.sam.firing_range = np.random.choice(sam_firing_range_list)
        self.sam.jammed_firing_range = self.sam.firing_range * self.jammer.jam_effectiveness

        sam_min_offset = \
            max(self.sam.firing_range, self.fighter.firing_range, self.jammer.jam_range) \
            + self.resolution * 2

        self.sam.offset = \
            sam_min_offset + np.random.random() * (self.sam.max_offset - sam_min_offset)
        self.sam.offset = np.round(self.sam.offset)

        condition = (self.sam.firing_range > self.jammer.jam_range) and \
                    (self.sam.firing_range > self.fighter.firing_range)

        if not condition:
            raise Exception('Error in condition_l1_generator')

    def condition_l2(self):
        sam_firing_range_list = np.arange(self.resolution * 3, self.jammer.jam_range, self.resolution)
        self.sam.firing_range = np.random.choice(sam_firing_range_list)
        self.sam.jammed_firing_range = self.sam.firing_range * self.jammer.jam_effectiveness

        fighter_firing_range_list = np.arange(0, self.sam.jammed_firing_range, self.resolution)
        self.fighter.firing_range = np.random.choice(fighter_firing_range_list)

        sam_min_offset = \
            max(self.sam.firing_range, self.fighter.firing_range, self.jammer.jam_range) \
            + self.resolution * 2

        self.sam.offset = \
            sam_min_offset + np.random.random() * (self.sam.max_offset - sam_min_offset)
        self.sam.offset = np.round(self.sam.offset)

        condition = (self.jammer.jam_range > self.sam.firing_range >
                     self.sam.jammed_firing_range > self.fighter.firing_range)

        if not condition:
            raise Exception('Error in condition_l2_generator')

    def reset_mission_condition(self):
        # Select the mission
        selection_list = self.mission_conditions  # ['w1','w2','w3']
        p = self.mission_probability
        cond_id = np.random.choice(selection_list, 1, p=p)[0]
        self.mission_condition = cond_id
        # print(f'\n--------- Mission condition: {cond_id}')

        func = 'self.condition_' + cond_id
        eval(func)()

    def reset(self):
        self.steps = 0
        self.mission_condition = None

        # Reset fighter
        self.reset_fighter()

        # Reset jammer
        self.reset_jammer()

        # Reset sam
        self.reset_sam()

        # Reset mission condition
        self.reset_mission_condition()

        # Reset render
        self.reset_render_fighter()
        self.reset_render_jammer()
        self.reset_render_sam()

        observation = self.get_observation()
        return observation

    def step(self, action_index):
        n = 0  # ローカル・ステップ
        done = False

        ''' アクション取得 '''
        self.fighter.action, self.jammer.action = self.ACTION_LIST[action_index]

        ''' 以前の状態を保存 '''
        self.fighter.previous_ingress = self.fighter.ingress
        self.jammer.previous_ingress = self.jammer.ingress
        self.jammer.previous_on = self.jammer.on

        while (done != True) and (n < self.action_interval):

            ''' 状態遷移(state transition)計算 '''
            self.fighter.ingress += self.fighter.action * self.fighter.speed * self.dt
            self.jammer.ingress += self.jammer.action * self.jammer.speed * self.dt

            # For the future application
            # rgb_shot = self.get_snapshot()

            # Jammerのjam_range内にSAMがいれば、Jammerが有効
            if self.jammer.ingress + self.jammer.jam_range >= self.sam.offset:
                self.jammer.on = 1

            # fighter win
            if (self.fighter.ingress + self.fighter.firing_range >= self.sam.offset):
                self.sam.alive = 0

            # fighter win with jammer
            if (self.jammer.on > 0.5) and \
                    (self.fighter.ingress + self.fighter.firing_range >= self.sam.offset):
                self.sam.alive = 0

            # clean sam win to fighter
            if (self.fighter.ingress >= self.sam.offset - self.sam.firing_range) and \
                    (self.jammer.on < 0.5):
                self.fighter.alive = 0

            # clean sam win to jammer
            if (self.jammer.ingress >= self.sam.offset - self.sam.firing_range) and \
                    (self.jammer.on < 0.5):
                self.jammer.alive = 0
                # self.jammer.on = 0

            # jammed sam win to fighter
            if (self.fighter.ingress >= self.sam.offset - self.sam.jammed_firing_range) and \
                    (self.jammer.on > 0.5):
                self.fighter.alive = 0

            # jammed sam win to jammer
            if (self.jammer.ingress >= self.sam.offset - self.sam.jammed_firing_range) and \
                    (self.jammer.on > 0.5):
                self.jammer.alive = 0
                # self.jammer.on = 0

            ''' 終了(done)判定 '''
            done = self.is_done()

            ''' 報酬(reward)計算 '''
            reward = self.get_reward(done)

            ''' ローカル・ステップのカウント・アップ '''
            n += 1

        ''' 観測(observation)の計算 '''
        observation = self.get_observation()

        ''' Set information, if any '''
        info = {}

        ''' for debug '''
        # pygame.time.wait(30)

        self.steps += 1

        return observation, reward, done, info

    def render_fighter(self):
        # transform coordinate from battle space to screen
        self.fighter.screen_x = self.fighter.ingress * self.to_screen_x - self.to_screen_offset

        # draw fighter
        screen_x = self.fighter.screen_x - self.fighter.radius
        screen_y = self.fighter.screen_y - self.fighter.radius
        self.screen.blit(self.fighter.surface, (screen_x, screen_y))

        # draw fighter's firing range
        range_screen_x = self.fighter.screen_x - self.fighter.screen_firing_range
        range_screen_y = self.fighter.screen_y - self.fighter.screen_firing_range
        self.screen.blit(self.fighter.surface_range, (range_screen_x, range_screen_y))

    def render_jammer(self):
        # transform coordinate from battle space to screen
        self.jammer.screen_x = self.jammer.ingress * self.to_screen_x - self.to_screen_offset

        # draw jammer
        screen_x = self.jammer.screen_x - self.jammer.radius
        screen_y = self.jammer.screen_y - self.jammer.radius
        self.screen.blit(self.jammer.surface, (screen_x, screen_y))

        # draw jammer's firing range
        range_screen_x = self.jammer.screen_x - self.jammer.screen_jam_range
        range_screen_y = self.jammer.screen_y - self.jammer.screen_jam_range
        self.screen.blit(self.jammer.surface_range, (range_screen_x, range_screen_y))

    def render_sam(self):
        # transform coordinate from battle space to screen
        self.sam.screen_x = self.sam.offset * self.to_screen_x - self.to_screen_offset

        # draw sam
        screen_x = self.sam.screen_x - self.sam.radius
        screen_y = self.sam.screen_y - self.sam.radius
        self.screen.blit(self.sam.surface, (screen_x, screen_y))

        if self.jammer.on < 0.5:
            # draw sam's firing range
            range_screen_x = self.sam.screen_x - self.sam.screen_firing_range
            range_screen_y = self.sam.screen_y - self.sam.screen_firing_range
            self.screen.blit(self.sam.surface_range, (range_screen_x, range_screen_y))
        else:
            # draw sam's jammed firing range
            range_screen_x = self.sam.screen_x - self.sam.screen_jammed_firing_range
            range_screen_y = self.sam.screen_y - self.sam.screen_jammed_firing_range
            self.screen.blit(self.sam.surface_jammed_range, (range_screen_x, range_screen_y))

    def render(self, mode='human'):
        if mode == 'human':
            self.screen.fill((0, 0, 0))
            if self.sam.alive > 0.5:
                self.render_sam()
            if self.jammer.alive > 0.5:
                self.render_jammer()
            if self.fighter.alive > 0.5:
                self.render_fighter()
            pygame.display.update()

            shot = pygame.surfarray.array3d(self.screen)
            shot = np.array(shot, dtype=np.uint8)
        else:
            shot = None
        return shot

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def get_observation(self):
        obs = []
        obs.append(self.fighter.ingress / self.sam.max_offset)
        obs.append(self.fighter.firing_range / self.fighter.max_firing_range)
        obs.append(self.jammer.ingress / self.sam.max_offset)
        obs.append(self.sam.offset / self.sam.max_offset)
        obs.append(self.sam.firing_range / self.sam.max_firing_range)
        obs.append(self.sam.jammed_firing_range /
                   (self.sam.max_firing_range * self.jammer.jam_effectiveness))

        obs.append(self.sam.alive)
        obs.append(self.jammer.on)

        observation = np.array(obs)  # (8,)
        return observation

    def get_reward_1(self, done):
        reward = 0
        step_reward_coef = 0.

        # For each time step
        fighter_x = self.sam.offset - self.fighter.ingress
        fighter_previous_x = self.sam.offset - self.fighter.previous_ingress
        fighter_dx = fighter_previous_x - fighter_x
        # reward = - 1 / self.max_steps
        # reward += np.sign(fighter_dx) * np.abs(fighter_dx) / (self.fighter.speed * self.dt) / self.max_steps
        reward += np.sign(fighter_dx) * np.abs(fighter_dx) / (self.fighter.speed * self.dt) * step_reward_coef

        jammer_x = self.sam.offset - self.jammer.ingress
        jammer_previous_x = self.sam.offset - self.jammer.previous_ingress
        jammer_dx = jammer_previous_x - jammer_x
        # reward = - 1 / self.max_steps
        # reward += np.sign(jammer_dx) * np.abs(jammer_dx) / (self.fighter.speed * self.dt) / self.max_steps
        reward += np.sign(jammer_dx) * np.abs(jammer_dx) / (self.fighter.speed * self.dt) * step_reward_coef

        # For done
        if done:
            if self.fighter.alive > .5:
                reward += 2
            if self.jammer.alive > .5:
                reward += 1
            if self.sam.alive > .5:
                reward -= 4

        return reward

    def get_reward_2(self, done):
        reward = 0

        # For done
        if done:
            if (self.fighter.alive > .5) and (self.jammer.alive > .5) and (self.sam.alive < .5):
                reward = 1
            else:
                reward = -1

        return reward

    def get_reward_3(self, done):
        reward = 0

        # For done
        if done:
            if (self.fighter.alive > .5) and (self.jammer.alive > .5) and (self.sam.alive < .5):
                reward = 1
            elif self.sam.alive > .5:
                reward = -1

        return reward

    def get_reward_4(self, done):
        reward = 0

        # For done
        if done:
            if (self.mission_condition == 'w1') and (self.fighter.alive > .5) and \
                    (self.jammer.alive > .5) and (self.jammer.on < .5) and (self.sam.alive < .5):
                reward = 1

            elif (self.mission_condition == 'w2') and (self.fighter.alive > .5) and \
                    (self.jammer.alive > .5) and (self.jammer.on > .5) and (self.sam.alive < .5):
                reward = 1

            elif (self.mission_condition == 'w3') and (self.fighter.alive > .5) and \
                    (self.jammer.alive > .5) and (self.jammer.on < .5) and (self.sam.alive < .5):
                reward = 1

            elif (self.mission_condition == 'l1') and (self.fighter.alive > .5) and (self.jammer.alive > .5):
                reward = 1

            elif (self.mission_condition == 'l2') and (self.fighter.alive > .5) and (self.jammer.alive > .5):
                reward = 1

            else:
                reward = -1

        return reward

    def get_reward_5(self, done):
        reward = 0

        # For jammer.on in w2
        if (self.mission_condition == 'w2') and (self.jammer.on > .5) and (self.jammer.previous_on < .5):
            reward = 1

        # For done
        if done:
            if (self.mission_condition == 'w1') and (self.fighter.alive > .5) and \
                    (self.jammer.alive > .5) and (self.jammer.on < .5) and (self.sam.alive < .5):
                reward = 1

            elif (self.mission_condition == 'w2') and (self.fighter.alive > .5) and \
                    (self.jammer.alive > .5) and (self.jammer.on > .5) and (self.sam.alive < .5):
                reward = 1

            elif (self.mission_condition == 'w3') and (self.fighter.alive > .5) and \
                    (self.jammer.alive > .5) and (self.jammer.on < .5) and (self.sam.alive < .5):
                reward = 1

            elif (self.mission_condition == 'l1') and (self.fighter.alive > .5) and (self.jammer.alive > .5):
                reward = 1

            elif (self.mission_condition == 'l2') and (self.fighter.alive > .5) and (self.jammer.alive > .5):
                reward = 1

            else:
                reward = -1

        return reward

    def get_reward(self, done):
        # reward = self.get_reward_1(done)
        # reward = self.get_reward_2(done)
        # reward = self.get_reward_3(done)
        reward = self.get_reward_4(done)
        # reward = self.get_reward_5(done)

        return reward

    def get_snapshot(self):
        shot = pygame.surfarray.array3d(pygame.transform.scale(self.screen, self.shot_shape))
        return np.array(shot, dtype=np.uint8)

    def is_done(self):
        done = False
        if (self.sam.alive < .5) or (self.jammer.alive < .5) or (self.fighter.alive < .5):
            done = True

        if (self.fighter.ingress <= self.fighter.allowable_negative_ingress) or \
                (self.jammer.ingress <= self.jammer.allowable_negative_ingress):
            done = True

        if self.steps > self.max_steps:
            done = True

        return done
