#!/usr/bin/env python

"""Minimal driver for running mupen64plus under program control.

This module replaces TensorKart's dependency on gym-mupen64plus. It
keeps the one piece that project provided that is hard to replace -
frame-synchronous controller input via the mupen64plus-input-bot plugin
- and drops everything else (headless Xvfb farms, config files, menu
navigation, the gym API).

How it works: the input-bot plugin (a C input plugin for mupen64plus)
makes an HTTP GET to ControllerServer once per emulated frame and the
response body is the controller state for that frame. send_controls()
therefore doubles as a frame-paced step: each action is held for
`frame_skip` frames before the call returns.

The emulator is launched headed (a normal window on your display) at
real-time speed, because the point is to watch it. Frames are captured
off the screen with mss, so keep the emulator window uncovered while
an agent is running.

Requires: mupen64plus, the input-bot plugin (built and installed, see
README), xdotool + xwininfo (to locate the window for capture).
"""

import json
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import mss
import numpy as np

# The emulator window is always launched at this resolution so frames
# line up with what the model was trained on
SCR_W = 640
SCR_H = 480

# Must match the defaults baked into the input-bot plugin (src/plugin.c)
DEFAULT_PORT = 8082

# Where `make install` puts the plugin (see mupen64plus-input-bot)
INPUT_BOT_PLUGIN = '/usr/local/lib/mupen64plus/mupen64plus-input-bot.so'


def launch_emulator(rom_path, input_plugin=None, extra_args=None):
    """Start mupen64plus in a normal window at real-time speed and
    return the process.

    input_plugin: path to an input plugin .so. Use INPUT_BOT_PLUGIN for
    agent control; leave as None for human play (the default SDL plugin
    reads your gamepad).
    """
    cmd = ['mupen64plus',
           '--nosaveoptions',
           '--resolution', f'{SCR_W}x{SCR_H}']
    if input_plugin is not None:
        cmd += ['--input', input_plugin]
    if extra_args:
        cmd += extra_args
    cmd.append(str(rom_path))

    print('Starting emulator: ' + ' '.join(cmd))
    return subprocess.Popen(cmd)


def find_emulator_window():
    """Locate the mupen64plus game window (X11 only, needs xdotool and
    xwininfo) so screenshots line up no matter where the window manager
    puts it. Returns (x, y) of the top-left corner, or None."""
    try:
        # there may be several matches (menus, dialogs) - the game
        # window is the one rendering at the capture resolution
        window_ids = subprocess.check_output(
            "xdotool search --onlyvisible --name 'Mupen64Plus'",
            shell=True, universal_newlines=True).split()

        for window_id in window_ids:
            info = subprocess.check_output(['xwininfo', '-id', window_id],
                                           universal_newlines=True)
            geometry = {}
            for line in info.splitlines():
                key, _, value = line.partition(':')
                key = key.strip()
                if key in ('Absolute upper-left X', 'Absolute upper-left Y', 'Width', 'Height'):
                    geometry[key] = int(value.strip())

            if (geometry.get('Width') == SCR_W and
                    geometry.get('Height') == SCR_H):
                return (geometry['Absolute upper-left X'],
                        geometry['Absolute upper-left Y'])
    except Exception:
        pass
    return None


def wait_for_emulator_window():
    """Block until the emulator window exists (it may not yet if the
    emulator was just started). Returns (x, y)."""
    if not shutil.which('xdotool'):
        print('xdotool not found - assuming the emulator window is at 0,0')
        return (0, 0)
    print('Waiting for the emulator window...')
    while True:
        window_pos = find_emulator_window()
        if window_pos is not None:
            print(f'Emulator window found at {window_pos}')
            return window_pos
        time.sleep(1)


class EmulatorWindow:
    """Screen capture of the emulator window. The window must stay
    visible and uncovered - anything on top of it ends up in the frame."""

    def __init__(self):
        self.offset_x, self.offset_y = wait_for_emulator_window()
        self._grabber = mss.MSS()

    def grab(self):
        """Capture the current frame as a (SCR_H, SCR_W, 3) RGB array."""
        image_array = \
            np.array(self._grabber.grab({"top": self.offset_y,
                                         "left": self.offset_x,
                                         "width": SCR_W,
                                         "height": SCR_H}),
                     dtype=np.uint8)

        # drop the alpha channel and flip red and blue channels (BGRA -> RGB)
        # (.copy() because np.flip returns a negative-stride view, which
        # breaks e.g. torch.from_numpy)
        return np.flip(image_array[:, :, :3], 2).copy()


class ControllerState(object):
    """One frame of N64 controller state. Field names are the JSON keys
    the input-bot plugin parses - do not rename them.

    Accepts a list of up to 16 values in this order (missing tail values
    default to 0):
    [ JX,  JY,  A,  B, RB, LB,  Z, CR, CL, CD, CU, DR, DL, DD, DU,  S]
    """

    NO_OP        = [  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    START_BUTTON = [  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]
    A_BUTTON     = [  0,   0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]

    def __init__(self, controls=NO_OP):
        controls = list(controls) + [0] * (16 - len(controls))
        self.X_AXIS = controls[0]
        self.Y_AXIS = controls[1]
        self.A_BUTTON = controls[2]
        self.B_BUTTON = controls[3]
        self.R_TRIG = controls[4]
        self.L_TRIG = controls[5]
        self.Z_TRIG = controls[6]
        self.R_CBUTTON = controls[7]
        self.L_CBUTTON = controls[8]
        self.D_CBUTTON = controls[9]
        self.U_CBUTTON = controls[10]
        self.R_DPAD = controls[11]
        self.L_DPAD = controls[12]
        self.D_DPAD = controls[13]
        self.U_DPAD = controls[14]
        self.START_BUTTON = controls[15]

    def to_json(self):
        return json.dumps(self.__dict__)


class ControllerServer(HTTPServer, object):
    """Serves controller state to the input-bot plugin.

    The plugin opens a new connection and GETs once per emulated frame;
    the response is the controller state for that frame. Each action
    passed to send_controls() is served for `frame_skip` frames, so a
    control loop that alternates send_controls() / capture runs in lock
    step with the emulator. frame_skip is read per request, so it can
    be changed on the fly (e.g. 1 for responsive manual input, more for
    an agent that thinks slower than the frame rate).
    """

    def __init__(self, port=DEFAULT_PORT, frame_skip=1):
        self.controls = ControllerState()
        self.controls_updated = threading.Event()
        self.response_sent = threading.Event()
        self.running = True
        self.responses_sent = 0
        self.frame_skip = frame_skip
        super(ControllerServer, self).__init__(('', port), self.ControllerRequestHandler)

        self._thread = threading.Thread(target=self.serve_forever, args=())
        self._thread.daemon = True
        self._thread.start()
        print(f'ControllerServer started on port {port}')

    def send_controls(self, controls):
        """Hold `controls` for frame_skip frames, then return."""
        self.responses_sent = 0
        self.controls = controls

        # Tell the request handler that the controls have been updated so it can send the response now:
        self.controls_updated.set()

        # Wait for response to actually be sent before returning:
        if self.running:
            self.response_sent.wait()
            self.response_sent.clear()

    def shutdown(self):
        self.running = False

        # Make sure we aren't blocking on anything:
        self.response_sent.set()
        self.controls_updated.set()

        # Shutdown the server:
        super().shutdown()
        super().server_close()

    class ControllerRequestHandler(BaseHTTPRequestHandler, object):

        def log_message(self, fmt, *args):
            pass

        def write_response(self, resp_code, resp_data):
            self.send_response(resp_code)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(resp_data.encode())

        def do_GET(self):
            # Wait for the controls to be updated before responding:
            if self.server.running:
                self.server.controls_updated.wait()

            if not self.server.running:
                # unblock the emulator's poll so it can exit cleanly
                self.write_response(500, "SHUTDOWN")
            else:
                ### respond with controller output
                self.write_response(200, self.server.controls.to_json())

            self.server.responses_sent += 1

            # If we have sent the controls 'n' times now...
            if self.server.responses_sent >= self.server.frame_skip:
                # ...we fire the response_sent event so the next action can happen:
                self.server.controls_updated.clear()
                self.server.response_sent.set()
