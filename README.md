TensorKart
==========

self-driving MarioKart with TensorFlow

Driving a new (untrained) section of the Royal Raceway:

![RoyalRaceway.gif](https://media.giphy.com/media/1435VvCosVezQY/giphy.gif)

Driving Luigi Raceway:

[![LuigiRacewayVideo](/screenshots/luigi_raceway.png?raw=true)](https://youtu.be/vrccd3yeXnc)

The model was trained with:
* 4 races on Luigi Raceway
* 2 races on Kalimari Desert
* 2 races on Mario Raceway

With even a small training set the model is sometimes able to generalize to a new track (Royal Raceway seen above).


Dependencies
------------
* `python` and `pip` then run `pip install -r requirements.txt`
* `mupen64plus` (install via apt-get)


Recording Samples
-----------------
1. Start your emulator program (`mupen64plus`) and run Mario Kart 64
2. Make sure you have a joystick connected and that `mupen64plus` is using the sdl input plugin
3. Run `record.py`
4. Make sure the graph responds to joystick input.
5. Position the emulator window so that the image is captured by the program (top left corner)
6. Press record and play through a level. You can trim some images off the front and back of the data you collect afterwards (by removing lines in `data.csv`).

![record](/screenshots/record_setup.png?raw=true)

Notes
- the GUI will stop updating while recording to avoid any slow downs.
- double check the samples, sometimes the screenshot is the desktop instead. Remove the appropriate lines from the `data.csv` file


Viewing Samples
---------------
Run `python utils.py viewer samples/luigi_raceway` to view the samples


Preparing Training Data
-----------------------
Run `python utils.py prepare samples/*` with an array of sample directories to build an `X` and `y` matrix for training. (zsh will expand samples/* to all the directories. Passing a glob directly also works)

`X` is a 3-Dimensional array of images

`y` is the expected joystick ouput as an array:

```
  [0] joystick x axis
  [1] joystick y axis
  [2] button a
  [3] button b
  [4] button rb
```


Training
--------
The `train.py` program will train a model using Google's TensorFlow framework and cuDNN for GPU acceleration. Training can take a while (~1 hour) depending on how much data you are training with and your system specs. The program will save the model to disk when it is done.


Play
----
The `play.py` program will use the [`gym-mupen64plus`](https://github.com/bzier/gym-mupen64plus) environment to execute the trained agent against the MarioKart environment. The environment will provide the screenshots of the emulator. These images will be sent to the model to acquire the joystick command to send. The AI joystick commands can be overridden by holding the 'LB' button on the controller.


Future Work / Ideas:
--------------------
* Add a reinforcement layer based on lap time or other metrics so that the AI can start to teach itself now that it has a baseline. The environment currently provides a reward signal of `-1` per time-step, which gives the AI agent a metric to calculate its performance during each race (episode), the goal being to maximize reward and therefore, minimize overall race duration.
* Could also have a shadow mode where the AI just draws out what it would do rather than sending actions. A real self driving car would have this and use it a lot before letting it take the wheel.
* Deep learning is all about data; perhaps a community could form around collecting a large amount of data and pushing the performance of this AI.


Special Thanks To
-----------------
* https://github.com/SullyChen/Autopilot-TensorFlow


Contributing
------------
Open a PR! I promise I am friendly :)
