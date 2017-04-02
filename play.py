#!/usr/bin/env python

from utils import resize_image, XboxController
import tensorflow as tf
import model
from termcolor import cprint
import gym
import gym_mupen64plus


# Play
class Actor:

    def __init__(self):
        # Start session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        # Load Model
        saver = tf.train.Saver()
        saver.restore(self.sess, "./model.ckpt")

        # Init contoller for manual override
        self.real_controller = XboxController()

    def get_action(self, obs):

        ### determine manual override
        manual_override = self.real_controller.LeftBumper == 1

        if not manual_override:

            ## Look
            vec = resize_image(obs)

            ## Think
            joystick = \
              model.y.eval(session=self.sess, feed_dict={model.x: [vec], model.keep_prob: 1.0})[0]

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

    env = gym.make('Mario-Kart-Luigi-Raceway-v0')
    obs = env.reset()
    env.render()
    print('env ready!')

    actor = Actor()
    print('actor ready!')

    print('beginning episode loop')
    total_reward = 0
    end_episode = False
    while not end_episode:
        action = actor.get_action(obs)
        obs, reward, end_episode, info = env.step(action)
        env.render()
        total_reward += reward

    print('end episode... total reward: ' + str(total_reward))

    obs = env.reset()
    print('env ready!')

    raw_input('press <ENTER> to quit')

    env.close()

