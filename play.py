#!/usr/bin/env python

"""Play: run the trained model against live Mario Kart 64.

Launches mupen64plus in a normal window at real-time speed with the
input-bot plugin, so you can watch the model drive. You drive the menus
yourself with the controller (the AI starts OFF) - line up a Time Trial
race, then tap LB to toggle the AI on/off.

Keep the emulator window uncovered: frames are captured off the screen.
"""

import torch
from termcolor import cprint

from emulator import (INPUT_BOT_PLUGIN, ControllerServer, ControllerState,
                      EmulatorWindow, launch_emulator)
from train import create_model, output_activation
from utils import resize_image, XboxController

ROM_PATH = 'MarioKart64.z64'

# Each AI action is held for this many emulated frames (the model was
# trained on ~200ms samples; 5 frames at 60fps is ~83ms). Manual input
# is served fresh every frame so driving the menus doesn't feel laggy.
AI_FRAME_SKIP = 5
MANUAL_FRAME_SKIP = 1


class Actor(object):

    def __init__(self):
        # Load in model from train.py and load in the trained weights
        self.model = create_model(pretrained=False)
        self.model.load_state_dict(torch.load('model_weights.pt', map_location='cpu'))
        self.model.eval()

        # Init contoller for manual override
        self.real_controller = XboxController()

        # The AI starts OFF (manual) - tap LB to toggle it on/off
        self.ai_enabled = False
        self._lb_was = 0

    def get_action(self, obs):

        ### toggle the AI on/off with LB (rising edge)
        lb = self.real_controller.LeftBumper
        if lb == 1 and self._lb_was == 0:
            self.ai_enabled = not self.ai_enabled
            cprint('AI ' + ('ON' if self.ai_enabled else 'OFF'), 'magenta')
        self._lb_was = lb

        manual_override = not self.ai_enabled

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
    actor = Actor()
    print('actor ready!')

    server = ControllerServer(frame_skip=MANUAL_FRAME_SKIP)
    emulator_process = launch_emulator(ROM_PATH, input_plugin=INPUT_BOT_PLUGIN)
    window = EmulatorWindow() # blocks until the emulator window exists

    print('ready - drive the menus, start a race, then tap LB to toggle the AI')

    try:
        obs = window.grab()
        while emulator_process.poll() is None:
            action = actor.get_action(obs)
            server.frame_skip = MANUAL_FRAME_SKIP if not actor.ai_enabled else AI_FRAME_SKIP
            server.send_controls(ControllerState(action))
            obs = window.grab()
        print('emulator closed')
    except KeyboardInterrupt:
        pass
    finally:
        print('shutting down...')
        server.shutdown()
        emulator_process.terminate()
