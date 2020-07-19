## Dependicies

1. Suitable Linux distribution e.g. Ubuntu
2. GCC or suitable C++ compiler
    * GCC download : https://gcc.gnu.org/install/download.html
3. OpenCV version 4.1.2 or later
    *   OpenCV download: https://opencv.org/releases/
4. Python 2.7
    * Python 2.7 install via terminal
        `sudo apt install python2`

## Usage

1. Clone repository:
    `git clone https://github.com/MusketMosez/ImageAnalyser`
2. cd into directory:
    `cd ImageAnalyser`
3. Make _out_ directory
    `mkdir out`
4. Change relevant directories in _ellipse.cpp_ on lines 672, 719 and 798.
   * e.g. '/home/user/workspace/ImageAnalyser/out'
5. Compile C++ file:
    `make`
6. Run the executable binary
    `./ellipse input.avi`
7. Capture region of interest of first frame using the mouse left click, then execute by using mouse right click
8. Run python script to create output video
    `python convert2vid.py`
 
