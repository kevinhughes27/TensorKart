#!/usr/bin/env python

"""Play: run the trained model against live Mario Kart 64.

Launches mupen64plus in a normal window at real-time speed with the
input-bot plugin, so you can watch the model drive. You drive the menus
yourself with the controller (the AI starts OFF) - line up a Time Trial
race, then tap LB to toggle the AI on/off.

Tap RB to toggle recording: frames, the applied inputs, and the model's
shadow predictions are saved to samples/<timestamp>_play/ in the same
format as record.py. Record yourself rescuing the AI when it fails -
those recovery samples are the most valuable training data there is
(DAgger). Then re-run prepare and train.

Keep the emulator window uncovered: frames are captured off the screen.
"""

import os
from datetime import datetime

import torch
from PIL import Image
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


class SampleRecorder:
    """Records (frame, applied input, model prediction) while playing.

    data.csv is written in the exact format record.py produces, so
    `python utils.py prepare samples/*` consumes play recordings
    unchanged. preds.csv additionally stores the model's shadow
    prediction for each frame - on manual frames this is what the AI
    *would* have done, so large |data - preds| marks exactly the
    recovery samples the model needs (DAgger).
    """

    def __init__(self, root='samples'):
        self.root = root
        self.enabled = False
        self._dir = None
        self._t = 0
        self._data_csv = None
        self._preds_csv = None

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self._dir = os.path.join(self.root, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_play')
            os.makedirs(self._dir)
            self._data_csv = open(os.path.join(self._dir, 'data.csv'), 'a')
            self._preds_csv = open(os.path.join(self._dir, 'preds.csv'), 'a')
            self._t = 0
        else:
            self._close()
        cprint(f'REC {"ON -> " + self._dir if self.enabled else "OFF"}', 'cyan')

    def record(self, obs, joystick, pred):
        if not self.enabled:
            return
        image_file = os.path.join(self._dir, f'img_{self._t}.png')
        Image.fromarray(obs).save(image_file)
        self._data_csv.write(image_file + ',' + ','.join(map(str, joystick)) + '\n')
        self._preds_csv.write(image_file + ',' + ','.join(map(str, pred)) + '\n')
        self._t += 1

    def _close(self):
        for f in (self._data_csv, self._preds_csv):
            if f is not None:
                f.close()
        self._data_csv = self._preds_csv = None

    def shutdown(self):
        if self.enabled:
            self.enabled = False
            self._close()


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

        # Recording starts OFF - tap RB to toggle it on/off
        self.recorder = SampleRecorder()
        self._rb_was = 0

    def predict(self, obs):
        """Run the model on a frame and return the joystick output."""
        ## Look
        vec = resize_image(obs)
        vec = torch.from_numpy(vec).permute(2, 0, 1).unsqueeze(0) # (66,200,3) -> (1,3,66,200)
        ## Think
        with torch.no_grad():
            return output_activation(self.model(vec))[0].numpy()

    def get_action(self, obs):
        """Returns (output, joystick, pred): the calibrated controller
        output to send, the normalized joystick values that were applied
        (the training label), and the model's prediction for this frame
        (None in manual mode - the main loop fills it in when recording
        so the model only runs on recorded frames)."""

        ### toggle the AI on/off with LB (rising edge)
        lb = self.real_controller.LeftBumper
        if lb == 1 and self._lb_was == 0:
            self.ai_enabled = not self.ai_enabled
            cprint('AI ' + ('ON' if self.ai_enabled else 'OFF'), 'magenta')
        self._lb_was = lb

        ### toggle recording on/off with RB (rising edge)
        rb = self.real_controller.RightBumper
        if rb == 1 and self._rb_was == 0:
            self.recorder.toggle()
        self._rb_was = rb

        manual_override = not self.ai_enabled

        if not manual_override:
            joystick = self.predict(obs)
            pred = joystick # the applied action IS the prediction

        else:
            joystick = self.real_controller.read()
            joystick[1] *= -1 # flip y (this is in the config when it runs normally)
            pred = None


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

        return output, joystick, pred


if __name__ == '__main__':
    actor = Actor()
    print('actor ready!')

    server = ControllerServer(frame_skip=MANUAL_FRAME_SKIP)
    emulator_process = launch_emulator(ROM_PATH, input_plugin=INPUT_BOT_PLUGIN)
    window = EmulatorWindow() # blocks until the emulator window exists

    print('ready - drive the menus, start a race, then tap LB to toggle the AI')

    try:
        obs = window.grab()
        frame = 0
        while emulator_process.poll() is None:
            action, joystick, pred = actor.get_action(obs)

            # Record at the AI's control cadence: every iteration while
            # the AI drives (each is AI_FRAME_SKIP emulated frames),
            # every AI_FRAME_SKIP-th iteration in manual mode (which
            # loops once per frame). Recorded frames and labels are
            # paired at the same rate the model acts at, so train and
            # play cadence match by construction.
            if actor.recorder.enabled and (actor.ai_enabled or frame % AI_FRAME_SKIP == 0):
                if pred is None:
                    pred = actor.predict(obs) # shadow prediction: what the AI would have done
                actor.recorder.record(obs, joystick, pred)

            server.frame_skip = MANUAL_FRAME_SKIP if not actor.ai_enabled else AI_FRAME_SKIP
            server.send_controls(ControllerState(action))
            obs = window.grab()
            frame += 1
        print('emulator closed')
    except KeyboardInterrupt:
        pass
    finally:
        print('shutting down...')
        actor.recorder.shutdown()
        server.shutdown()
        emulator_process.terminate()
