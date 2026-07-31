#!/usr/bin/env python

from utils import resize_image, XboxController
from termcolor import cprint

import gymnasium as gym
import gym_mupen64plus
from train import create_model, output_activation
import torch

# Play
class Actor(object):

    def __init__(self):
        # Load in model from train.py and load in the trained weights
        self.model = create_model(pretrained=False)
        self.model.load_state_dict(torch.load('model_weights.pt', map_location='cpu'))
        self.model.eval()

        # Init contoller for manual override
        self.real_controller = XboxController()

    def get_action(self, obs):

        ### determine manual override
        manual_override = self.real_controller.LeftBumper == 1

        if not manual_override:
            ## Look
            vec = resize_image(obs)
            vec = torch.from_numpy(vec).permute(2, 0, 1).unsqueeze(0) # (66,200,3) -> (1,3,66,200)
            ## Think
            with torch.no_grad():
                joystick = output_activation(self.model(vec))[0].numpy()

        else:
            joystick = self.real_controller.read()
            joystick[1] *= -1 # flip y (this is in the config when it runs normally)


        ## Act

        ### calibration
        output = [
            int(joystick[0] * 80),
            int(joystick[1] * 80),
            int(round(joystick[2])),
            int(round(joystick[3])),
            int(round(joystick[4])),
        ]

        ### print to console
        if manual_override:
            cprint("Manual: " + str(output), 'yellow')
        else:
            cprint("AI: " + str(output), 'green')

        return output


if __name__ == '__main__':
    # Which track to race and who to drive as.
    # Courses: Luigi-Raceway, Moo-Moo-Farm, Koopa-Troopa-Beach, Kalimari-Desert,
    #          Toads-Turnpike, Frappe-Snowland, Choco-Mountain, Mario-Raceway,
    #          Wario-Stadium, Sherbet-Land, Royal-Raceway, Bowsers-Castle,
    #          DKs-Jungle-Parkway, Yoshi-Valley, Banshee-Boardwalk, Rainbow-Road
    # Characters: mario, luigi, peach, toad, yoshi, d.k., wario, bowser
    COURSE = 'Luigi-Raceway'
    CHARACTER = 'mario'

    # Whether to open a second window showing the agent's view. Off by
    # default: you can watch the emulator itself, and another window on
    # top of it would corrupt the screen capture.
    RENDER = False

    env = gym.make(f'Mario-Kart-{COURSE}-v0', character=CHARACTER)

    obs, info = env.reset()
    if RENDER:
        env.render()
    print('env ready!')

    actor = Actor()
    print('actor ready!')

    print('beginning episode loop')
    total_reward = 0
    end_episode = False
    while not end_episode:
        action = actor.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        end_episode = terminated or truncated
        if RENDER:
            env.render()
        total_reward += reward

    print('end episode... total reward: ' + str(total_reward))

    obs, info = env.reset()
    print('env ready!')

    input('press <ENTER> to quit')

    env.close()
