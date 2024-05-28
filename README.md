# Video Thumbnail Generator

Automatically generate thumbnail grid images from video files. 
User can provide one or more video file paths as positional arguments and the program will then extract a set of representative thumbnails from each video, 
along with key metadata information such as the video title, duration, size, bitrate, resolution and fps. 
The thumbnails will have the corresponding timestamp displayed on the upper left corner and are arranged in a grid layout with the number of rows and columns configurable through the --rows and --cols options (default is 5 rows and 7 columns). 
The program also offers an --overwrite option, which allows users to overwrite any existing thumbnail images if needed. 


## Arguments
```
positional arguments:
  videospaths           Video paths to generate thumbnails

options:
  -h, --help            show this help message and exit
  --rows ROWS, -r ROWS  Number of rows in output image grid (default: 5)
  --cols COLS, -c COLS  Number of columns in output image grid (default: 7)
  --overwrite, -o       Overwrite existent thumbnails (default: False)
```
## Requirements
```
Python 3

numpy==1.23.3
opencv_contrib_python==4.6.0.66
tqdm==4.66.1
```

## Examples and usage

Example 1: 

Grid of 7x5

```
py thumbnail-generator.py "New quantum computers - Potential and pitfalls.mp4" --rows 5 --cols 7
```
![Example Image 1](https://github.com/leandroesposito/thumbnail-generator/blob/main/examples/New%20quantum%20computers%20-%20Potential%20and%20pitfalls_snapshot.jpg "Example Image 1")
---

Example 2: 

Grid of 10x30

```
py thumbnail-generator.py "Traveling Ecuador by train | DW Documentary.mp4" --rows 30 --cols 10
```
![Example Image 2](https://github.com/leandroesposito/thumbnail-generator/blob/main/examples/Traveling%20Ecuador%20by%20train%20%EF%BD%9C%20DW%20Documentary_snapshot.jpg "Example Image 2")
