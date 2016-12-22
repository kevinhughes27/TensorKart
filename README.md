MarioKartAI
===========

ToDo
----
* finish train.py with TensorFlow
* implement play.py


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

Your window should look like this:

6. Press record and play through a level. You can trim some samples off the front and back of the data you collect afterwards.

Note - the GUI will stop updating while recording to avoid any slow downs.


Viewing Samples
---------------
Run `python viewer.py samples/luigi_raceway` to view the samples
