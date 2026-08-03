TensorKart
==========

self-driving MarioKart with PyTorch

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
* `python` and [`uv`](https://docs.astral.sh/uv/) then run `uv sync`
* `mupen64plus` (install via apt-get)
* for `play.py`: [`mupen64plus-input-bot`](https://github.com/kevinhughes27/mupen64plus-input-bot) built and installed (`make all && sudo make install`) - the input plugin that lets a program drive the emulator
* `xdotool` and `xwininfo` (to locate the emulator window for screen capture)


Recording Samples
-----------------
1. Make sure you have a joystick connected (Xbox and wired PS4 controllers both work - on a PS4 controller square is the B button)
2. Run `record.py` - it starts `mupen64plus` for you in a window (with the default SDL input plugin, so your controller drives the game)
3. Make sure the graph responds to joystick input.
4. If `xdotool` is installed the program will find the emulator window automatically. Otherwise position the emulator window so that the image is captured by the program (top left corner)
5. Press record and play through a level. You can trim some images off the front and back of the data you collect afterwards (by removing lines in `data.csv`).

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

`episodes` is the recording-session name per frame, so training can hold out whole episodes for validation (frames from one race are near-duplicates; a random split would leak them into both train and val).


Training
--------
The `train.py` program will train a model using PyTorch and [timm](https://github.com/huggingface/pytorch-image-models). Instead of training from scratch, the model uses transfer learning: a MobileNetV3 backbone pretrained on ImageNet with a small linear head for the joystick output. Training runs in two phases - first only the head is trained with the backbone frozen, then the whole model is fine-tuned with a lower learning rate. The program will save the best model from all epochs of training to disk when it is done.

Validation is a held-out episode (`VAL_EPISODES` in train.py), not a random split. Training data is augmented on the fly with horizontal flips (steering negated - this also balances the left-heavy recorded tracks) and brightness/contrast jitter.

`eval.py` scores the trained model against the recorded samples, with a per-episode breakdown and a |model - human| disagreement histogram (the tail of that histogram is where the model needs more data).
 

Play
----
Run `play.py`. It starts `mupen64plus` in a window at real-time speed using the input-bot plugin, which polls the `ControllerServer` in `emulator.py` for controller state once per emulated frame - so the agent loop runs in lock step with the emulator. The AI starts off: drive the menus yourself with the controller (start a Time Trial race), then tap 'LB' to toggle the AI on/off. Frames are captured off the screen, so keep the emulator window uncovered while the AI is driving. The emulator is launched in a normal window at real-time speed on purpose: the point is to watch the model drive.

Tap 'RB' to toggle recording. Frames, applied inputs, and the model's shadow predictions (what the AI *would* have done) are saved to `samples/<timestamp>_play/` in the same format as `record.py` - re-run prepare and train to fold them in. Recording yourself rescuing the AI when it fails is the point: behavior cloning only learns states the human visits, so it has no idea how to recover once it drifts off line. Expert corrections in states the *model* visits are exactly the data that fixes this (DAgger). Frames are recorded at the AI's control cadence, so train and play cadence match by construction.


Future Work / Ideas:
--------------------
* Temporal model: stack the last few frames (or a recurrent head) so the model can perceive speed and rotation - a single frame is ambiguous about how much steering a turn needs.
* Architecture experiments, judged against the held-out episode: a spatial-softmax head (classification backbones global-pool away the "where" information steering needs) and a small from-scratch CNN as a baseline challenger.
* Add a reinforcement layer based on lap time or other metrics so that the AI can start to teach itself now that it has a baseline. Two promising ingredients: mupen64plus save states for instant reset to race start (the core exposes `M64CMD_STATE_LOAD` and the input-bot plugin already has a handle to the core library), and [`stable-retro`](https://github.com/Farama-Foundation/stable-retro) (maintained gym-retro fork with an N64 core: save states, RAM reads for reward, frame-exact headless stepping) if RL ever becomes the focus.
* Deep learning is all about data; perhaps a community could form around collecting a large amount of data and pushing the performance of this AI.


Related Projects:
--------------------
[`Xbox Game AI`](https://github.com/mgagvani/Xbox-Game-AI) - Uses [`PYXInput`](https://github.com/bayangan1991/PYXInput) for direct control of any Xbox/PC game.

[`SerpentAI`](https://github.com/SerpentAI/SerpentAI) - Game Agent Framework to create AIs for any game.

[`Donkey Gym`](https://github.com/tawnkramer/gym-donkeycar) - OpenAI Gym Environments for self-driving "[`Donkey Car`](https://github.com/autorope/donkeycar)". 

[`AirSim`](https://github.com/microsoft/AirSim) - An Unreal Engine simulator for autonoumous vehicles.

Special Thanks To
-----------------
* https://github.com/SullyChen/Autopilot-TensorFlow


Contributing
------------
Open a PR! I promise I am friendly :)
