#!/usr/bin/env python

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from utils import take_screenshot, prepare_image
from utils import XboxController
import tensorflow as tf
import model
from termcolor import cprint

PORT_NUMBER = 8082

# Start session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Load Model
saver = tf.train.Saver()
saver.restore(sess, "./model.ckpt")

# Init contoller for manual override
real_controller = XboxController()

# Play
class myHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        ## Look
        bmp = take_screenshot()
        vec = prepare_image(bmp)

        ## Think
        joystick = model.y.eval(feed_dict={model.x: [vec], model.keep_prob: 1.0})[0]

        ## Act
        ### manual override
        manual_override = real_controller.manual_override()

        if (manual_override):
            joystick = real_controller.read()
            joystick[1] *= -1 # flip y (this is in the config when it runs normally)

        ### calibration
        output = [
            int(joystick[0] * 80),
            int(joystick[1] * 80),
            int(round(joystick[2])),
            int(round(joystick[3])),
            int(round(joystick[4])),
        ]

        ### print to console
        if (manual_override):
            cprint("Manual: " + str(output), 'yellow')
        else:
            cprint("AI: " + str(output), 'green')

        ### respond with action
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(output)
        return


if __name__ == '__main__':
    server = HTTPServer(('', PORT_NUMBER), myHandler)
    print 'Started httpserver on port ' , PORT_NUMBER
    server.serve_forever()
