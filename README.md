Webcam demo for gender conversion using DiscoGAN
=========================================

This repo uses the default webcam for live gender conversion.

The uploaded models were trained for ~200,000 iterations using the facescrub dataset.

![Recording](https://github.com/ptrblck/DiscoGAN/raw/master/assets/live_demo_recording.gif "Recording")



Additional Prerequisites
-------------
   - Pygame

Usage
-----
```
usage: run_inference.py [-h] [--modelA MODELA] [--modelB MODELB]
                        [--face_cascade FACE_CASCADE] [--device DEVICE]
                        [--size SIZE] [--output OUTPUT] 
```
The currently used and uploaded models are trained on the facescrub dataset for gender conversion 
(modelA: female --> male, modelB: male --> female).

To start the demo with the default models, cd into `./discogan` and call the script `run_inference.py`

If you trained another DiscoGAN, just copy your model files to the appropriate folder in `./models` and set the model path parameters (`--modelA` and `--modelB`) accordingly.

Currently the default opencv haarcascade for frontal faces is used to detect your face.

Try to use another face detector or try to smooth the detection, when the results are too noisy.

You can also adjust the webcam resolution defined in `run_inference.py` as `CAPTURE_SIZE`. The default values is set to 640x480.


Run the demo
----------------
During the demo you can use the following keys:
   - ESC: close application
   - s: switch between both generators (male -> female, female -> male)
   - p: pause application
   - r: start recording. Each frame will be saved in the format (`%03d.jpg % counter`) under `./recording/`. Note: Old frames will be overwritten.

You can easily convert the recorded frames to a .gif with ```convert -delay 6 -loop 0 *.jpg myimage.gif```.

Don't forget to set the right delay param based on your FPS. See [ImageMagick Delay param](http://www.imagemagick.org/script/command-line-options.php#delay) for more information.

Using a GTX 1080, I could achieve ~15 FPS, while using my laptop with CPU only (i3-2310M CPU @ 2.10GHz) yields ~4 FPS.


Issues
------
I hade to comment out `import ipdb` in model.py, since it threw an error while developing of this demo.
