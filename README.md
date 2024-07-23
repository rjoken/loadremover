# loadremover
Load remover for YGO DotR Speedrun videos

A small python script used for the removal of load times from YGO: DotR speedrun videos. The script uses OpenCV to detect when a frame of video contains the 'card' sprite, and removes that frame from the video, allowing you to get the run time without loads.

![gcahout](https://github.com/user-attachments/assets/29b73f36-c36f-41e9-8e76-91b0f9ee223c)

The above image is the output for `loadremover-image.py`, which can be used for troubleshooting the loading image detection. Run this by invoking the script with `python loadremover-image.py in.png out.png` with in.png being your input file and out.png being your output file. The card sprite image being detected is `loading.png` in the directory of the script.

The load removal for a video file is run in the same way; `python loadremover.py in.mp4 out.mp4`.

Requires python 3, sys, getopt, cv2, ffmpegcv and numpy to be installed via pip.
