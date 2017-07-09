#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:52:06 2017

@author: pbialecki
"""
from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np

import os, errno
import time
import argparse

import cv2

import pygame
import pygame.camera
from pygame.locals import KEYDOWN, K_ESCAPE, K_s, K_p, K_r, QUIT

#Globals
FLAGS = None


def run():
    # Set options
    IMAGE_SIZE = 64  # Fixed, since the DiscoGAN was trained on this size
    CAPTURE_SIZE = (640, 480)  # Adjust for your webcam
    FONT_SIZE = 15  # Adjust for better visibility

    # Load models
    if torch.cuda.is_available():
        generator_A = torch.load(FLAGS.modelA).cuda()
        generator_B = torch.load(FLAGS.modelA).cuda()
    # Since the models were trained on GPU we have to remap them to CPU
    else:
        generator_A = torch.load(
            FLAGS.modelA, map_location=lambda storage, loc: storage)
        generator_B = torch.load(
            FLAGS.modelB, map_location=lambda storage, loc: storage)

    # Create OpenCV Face detector
    face_cascade = cv2.CascadeClassifier(FLAGS.face_cascade)

    # Create Capture object
    pygame.init()
    pygame.camera.init()
    display = pygame.display.set_mode((FLAGS.size * 2, FLAGS.size), 0)
    camera = pygame.camera.Camera(FLAGS.device, CAPTURE_SIZE)

    # Try to open default capture device
    try:
        camera.start()
    except SystemError as e:
        print(e.message)
        exit(errno.ENOENT)

    snapshot = pygame.surface.Surface(CAPTURE_SIZE, 0, display)
    font = pygame.font.SysFont("monospace", FONT_SIZE)

    # Switch for generator
    generator = 1
    generators = [generator_A, generator_B]
    # Change the text if your GAN is trained with a different class set
    info_text = ['female -> male', 'male -> female']

    # Save to local variable to performance issues
    draw_size = (FLAGS.size, FLAGS.size)

    # Start loop
    counter = 0
    going = True
    pause = False
    recording = False
    while going:
        # Get keyboard input
        events = pygame.event.get()
        for e in events:
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                going = False
            elif (e.type == KEYDOWN and e.key == K_s):
                # Switch generator
                generator ^= 1
            elif (e.type == KEYDOWN and e.key == K_p):
                pause = not pause
            elif (e.type == KEYDOWN and e.key == K_r):
                recording = not recording

        # Process
        if not pause:
            t0 = time.time()

            snapshot = camera.get_image(snapshot)
            image = pygame.surfarray.array3d(snapshot)
            image = np.rot90(image, -1)

            # Find face
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
            # Skip, if no faces were found
            if not isinstance(faces, np.ndarray):
                continue

            # Cut only first face
            (x, y, w, h) = faces[0]
            roi_color = image[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color, (IMAGE_SIZE,
                                               IMAGE_SIZE))  # Resize for net
            image_face = np.array(roi_color).astype(np.float32) / 255.  # Scale

            image_face = image_face[:, :, [2, 1, 0]]  # Swap channels (to BGR)

            # Create pytorch Variable
            image_face = image_face.transpose(2, 0, 1)  # Transpose
            image_face = image_face[np.newaxis, ...]
            A = Variable(torch.FloatTensor(image_face))
            if torch.cuda.is_available():
                A = A.cuda()

            # Run inference
            gen_AB = generators[generator](A)

            # copy back to CPU and visualize results
            gen_AB_image = gen_AB.data.cpu().numpy()
            gen_AB_image = gen_AB_image[0, ::-1, ...].transpose(1, 2, 0)

            # Draw image
            gen_AB_image = np.rot90(gen_AB_image)
            gen_snapshot = pygame.surfarray.make_surface(gen_AB_image * 255.)
            face_snapshot = pygame.surfarray.make_surface(np.rot90(roi_color))

            # Scale
            face_snapshot = pygame.transform.scale(face_snapshot, draw_size)
            gen_snapshot = pygame.transform.scale(gen_snapshot, draw_size)

            display.blit(face_snapshot, (0, 0))
            display.blit(gen_snapshot, (draw_size[0], 0))

            fps = 1 / (time.time() - t0)
            font_color = (250, 250, 250)
            label = font.render("{0:.1f} FPS".format(fps), 1, font_color)
            display.blit(label, (0, 0))
            info = font.render(info_text[generator], 1, font_color)
            display.blit(info, (FLAGS.size, 0))
            frame_text = font.render('%d' % counter, 1, font_color)
            display.blit(frame_text, (draw_size[0] * 2 - 2 * FONT_SIZE,
                                      draw_size[1] - 2 * FONT_SIZE))

            pygame.display.flip()

            if recording:
                pygame.image.save(display,
                                  os.path.join(FLAGS.output,
                                               '%03d.jpg' % counter))

            counter = counter + 1

    # Shutdown
    camera.stop()
    pygame.quit()


def main():
    if not os.path.exists(FLAGS.modelA):
        print('Could not find model A in {}'.format(FLAGS.modelA))
        exit(errno.ENOENT)
    if not os.path.exists(FLAGS.modelB):
        print('Could not find model B in {}'.format(FLAGS.modelB))
        exit(errno.ENOENT)
    if not os.path.exists(FLAGS.face_cascade):
        print('Could not find opencv face haarcascade in {}'.format(
            FLAGS.face_cascade))
        exit(errno.ENOENT)
    if not os.path.exists(FLAGS.output):
        os.mkdir(FLAGS.output)
    run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--modelA',
        type=str,
        default=os.path.join('..', 'models', 'facescrub', 'discogan', 'model_gen_A-20'),
        help='Path to model file')
    parser.add_argument(
        '--modelB',
        type=str,
        default=os.path.join('..', 'models', 'facescrub', 'discogan', 'model_gen_B-20'),
        help='Path to model file')
    parser.add_argument(
        '--face_cascade',
        type=str,
        default=os.path.join('.', 'haarcascade_frontalface_default.xml'),
        help='Path to opencv face haarcascade')
    parser.add_argument(
        '--device', type=str, default='/dev/video0', help='Capture device')
    parser.add_argument(
        '--size',
        type=int,
        default=256,
        help='Output size for visualization in pixel')
    parser.add_argument(
        '--output',
        type=str,
        default=os.path.join('..', 'recording'),
        help='Output directory for recordings')
    FLAGS, unparsed = parser.parse_known_args()
    main()
