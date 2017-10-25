# Tracking via Template Matching (with CUDA) 
## Description
    To demonstrate my skills with CUDA I decided to implement a template 
    matching tracking algorithm. On each new frame, the tracker takes the 
    detected car ROI from the last frame and template matches it in the new 
    frame. The pixel location with the best match score becomes the new origin
    for the car's bounding box.

    Some glaring issues with my implementation are that the size of the 
    bounding box region never changes and it is not robust to occlusions, 
    however it works pretty well in the beginning of the sequence and is a 
    nice function to implement with GPU parallelization.

## Future work
    With more time, I would have liked to implement a reduction algorithm for
    finding the location of the minimum score value in the array produced by 
    template matching. Right now, the function implemented takes longer to run
    than all other pieces of the code. Another opportunity for speeding up the 
    code is to not have the device memory allocate/free on every new frame. A 
    better approach would be to simply overwrite the old data instead, since 
    the image sizes don't change.

## Compiling
    Unfortunately I'm not familiar with building my own CUDA projects on Linux,
    so I built this tracker project in Visual Studio 2013. I've 
    included a pre-built *.exe* file as well as all my source code and VS project
    files, so hopefully that's enough to get it going. (I built for Release x64.)
    
## Dependencies
Libs used:
* opencv_core310.lib
* opencv_highgui310.lib
* opencv_imgproc310.lib
* opencv_imgcodecs310.lib

CUDA
* Version 7.5
