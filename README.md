Webcam demo for gender conversion using DiscoGAN
=========================================

This repo uses the default webcam for live gender conversion.
The uploaded models were trained for ~200,000 iterations using the facescrub dataset.

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
The currently used models are trained on the facescrub dataset for gender conversion (modelA: female -> male, modelB: male -> female).
Just add your trained generators to the `/models/` path and set the parameters while calling `run_inference.py`

Currently the default opencv haarcascade for frontal faces is used to detect your face.
Try to use another face detector or try to smooth the detection, when the results are too noisy.
Also you can try to adjust your webcam resolution defined in `run_inference.py` with `CAPTURE_SIZE`.


Run the demo
----------------
Use the following keys for:
    - ESC: close application
    - s: switch between both generators (male -> female, female -> male)
    - p: pause application
    - r: start recording. Each frame will be saved as `%03d.jpg % counter` in `./recording/`. Old frames will be overwritten.

You can easily convert the recorded frames to a .gif with `convert -delay 6 -loop 0 *.jpg myimage.gif`.
Don't forget to set the right delay param based on your FPS. See [ImageMagick Delay param](http://www.imagemagick.org/script/command-line-options.php#delay) for more information.

Using a GTX 1080, I could achieve ~15 FPS, while using my laptop with CPU only (i3-2310M CPU @ 2.10GHz) yields ~4 FPS.


Issues
------
I hade to comment out `import ipdb` in model.py, since it threw an error while developing of this demo.
