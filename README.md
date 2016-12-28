TensorKart
==========

self-driving MarioKart with TensorFlow

Dependencies
------------
* `python` and `pip` then run `pip install -r requirements.txt`
* `mupen64plus` (install via apt-get)


Recording Samples
-----------------
1. Start your emulator program (`mupen64plus`) and run Mario Kart 64
2. Make sure you have a joystick connected
3. Run `record.py`
4. Make sure the graph responds to joystick input. It will look slow but that is because the sample rate is low on purpose.
5. Position the emulator window so that the image is captured by the program
6. Press record and play through a level. You can trim some samples off the front and back of the data you collect afterwards.

![record](/screenshots/record_setup.png?raw=true)

Note - the GUI will stop updating while recording to avoid any slow downs.


Viewing Samples
---------------
Run `python viewer.py samples/luigi_raceway` to view the samples


Preparing Training Data
-----------------------
The `prepare.py` script takes an array of sample directories as arguments and builds an `X` and `y` matrix for training. (zsh will expand samples/* to all the directories)

`X` is a 2-Dimensional array where each row is a flattened image. (each cell is therefore a unsigned int)

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
The `train.py` program will train a model using Google's TensorFlow framework and cuDNN for GPU acceleration. Training can take a while (~1 hour) depending on how much data you are training with. The program will save the model to disk when it is done.


Play
----
The `play.py` program will take screenshots of your desktop expecting the emulator to be in the top right corner again. These images will be sent to the model to acquire the joystick command to send.

Note - you need to start the emulator a custom input driver in order to pass the output from the AI to the emulator:

```
mupen64plus --input ~/src/mupen64plus-input-bot/mupen64plus-input-bot.so MarioKart64.z64
```

current status:
* the emulator is using this controller for player 1
* the emulator does not respond to the inputs I send
* `jstest-gtk` does respond to these inputs
* `evtest` on my fake controller does not respond either
  * emit_click does showup in evtest

* **Attempt to just script mario kart to get this working**


Does it Generalize?
-------------------
* The network should be able to replay a level since at this point its can be overfitted and pretty much rememeber a sequence of commands.
  * That is unless it gets stuck and drives off the course and can't get back
* Try it on a new level and see what it does. This is in place of a validation data set.


ToDo
----
* drop pygame and use https://pypi.python.org/pypi/inputs (looks lighter weight than pygame)
* I could/should update my model to this one https://github.com/SullyChen/Autopilot-TensorFlow
* possibly switch to https://keras.io/ (a nice wrapper on top of TensorFlow)

* I could simplify data recording to just the axis (aka steering angle predictor)

* I am getting close to too much data to hold in memory

* record new data using as much auto as possible - the idea being to get specific samples of me fixing the AI when it gets stuck

* Mario raceway could be a good track

* [he](https://github.com/SullyChen/Autopilot-TensorFlow) loads his batches sequentially. I wonder if this helps? It would be minimizing a logical sequence rather than a random one.

Future:
-------
* could also have a shadow mode where the AI just draws out what it would do rather than sending actions. A real self driving car would have this and use it a lot before letting it take the wheel.
