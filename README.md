MarioKartAI
===========

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


Preparing Training Data
-----------------------

The `prepare.py` script takes an array of sample directories as arguments and builds an `X` and `y` matrix for training.

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
* finish train.py with TensorFlow


Play
----
* load model
  ```python
  # ... other variables need to be setup still
  saver = tf.train.Saver()
  saver.restore(sess, "model.ckpt")
  ```
* acquire screenshot and send to tensorflow
* send the output from tensorflow as the joystick input


Does it Generalize?
-------------------
* The network should be able to replay a level since at this point its can be overfitted and pretty much rememeber a sequence of commands.
  * That is unless it gets stuck and drives off the course and can't get back
* Train on several levels of Mario Kart
* Try it on a new level and see what it does.


Notes
-----
* I could also always pass the previous frame as part of the input. Or is this better solved with a different type of network
* update record to use https://pypi.python.org/pypi/inputs (looks lighter weight than pygame)
