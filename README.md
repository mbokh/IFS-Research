# Quick Start Usage
This repo contains code to process hyperspectral images/videos as produced by the SHEAR camera system. 

This software was developed in Python 3.7, but any reasonably close version should suffice. Additionally, it has dependencies on `opencv-python`, `matplotlib`, `numpy`, `scipy`, as well as their own dependencies

In order to begin the processing, put the hyperspectral video (or a folder containing the frames) in the `sources` folder. The only scripts you need to directly interact with are in the top-level directory and are all prefixed with `main`. Open the script `mainExtractData.py` and edit this line: `video = VideoSource.VideoSource(filename="ENTER FILE NAME HERE", skip=0, end=-1, spectraStart=150, spectraEnd=1023, flipLR=True)` to put in the name of the video file in the filename argument.  In addition, specify the x range over which the spectral lines appear. These values do not have to particularly exact, they are simply used to isolate certain regions of the frames to look for either spectral lines or particles. Do not edit the `flipLR=True` parameter.

Running this script will create a pickle file which contains all the useful information extracted from the video. Depending on how long / complex the video is, extraction could take over an hour, but it will print out its progress periodically. 

After the pickle file is created, there is some flexibility to use the data in whatever way is useful. The `mainCreateCSV.py` script can be run, which will convert the pickle file into a csv file which can be used for processing outside of Python or this repository. The format for the csv is 
```
minWavelength, maxWavelength
particleId, frameNumber, temperature, [spectraArray], spectraCode, [boundingBox], brightness, occlusionCount
...
```

You can also run the `mainCreateCompiledVideo.py` script, which will render a video showing physical-space particles, color-coded bounding boxes "corners", a graph of temperature and spectra in time. Due to the large number of particles in the video, this is not intended to provide any substantial quantitative information for further research, nor does it incorporate all of the extracted information. It is simply as a way to conveniently view data and have a "big picture" view into the data.

Running `mainCreateSpectraVideo.py` will create a video showing the spectral information for each particle, including the regressed temperature and its corresponding curve.

All the extracted data (pickle files, video files, csv files), will appear in the `extractedData` folder.

# Documentation
Although some object-oriented design efforts are lost on Python, I tried to reasonably structure my code to follow those principles, which would facilitate porting to a different and faster language in the future. 

### Sources
In order to provide an abstraction layer over the actual data source, there are two files in the `Sources` folder, `VideoSource` and `FramesSource`, which wrap a video file and folder of frames respectively. These two files have exactly the same method prototypes except for the constructor, so they are totally interchangeable. When creating a `VideoSource`, simply supply the name of the video file in the `Sources` folder, but free of any path names. A `FramesSource` will expect image names in a particular format: for a folder named "FOLDER", it will look for frames inside the folder named "FOLDER000001.tif", with the frame number counting up. When creating a `FramesSource` object, supply the folder name "FOLDER" in the argument called `prefix`. If your data format varies somewhat from this default format, it should be very simple to change the code to match your format. 

The other arguments of the `VideoSource` and `FramesSource` object supply additional information about the video. `skip` lets you skip the first n frames, in case nothing interesting happens then. In the extracted data, all the frameNums are relative to this initial offset. Similarly, `end` allows you to stop the processing at a given frame number, which you can set to -1 if you want to process to the end. `spectraStart` and `spectraEnd` supply the x positions of the region over which spectral lines are present. These values do not need to be extremely accurate, but they should easily partition the regions containing the particles and the spectral lines. It is also recommended that the other value be at the edge of the screen. For example:
```
-----------------------------------x->   
|            |                       |
|            |                       |
y  particles |    spectra lines      |
|            |                       |
V            |                       |
--------------------------------------
        spectraStart           spectraEnd
``` 

### Particle Detection
The videos produced from the camera system are at a very high frame rate, so each frame has very little exposure time and only effectively captures the very bright burning particles of interest. This means that the background of the frame is essentially all black, making particle detection relatively simple. 

First, I apply a median blur to remove the little bit of white speckle noise that does appear. Then, I apply an unsharp mask to help remove some of the gradient effects from the image, which will aid in segmentation. I then use a simple global binary threshold and then examine the connected components. Components with a bounding box area of 2 or less, I filter out as this could still be noise, make the processing a lot slower, and probably doesn't contain very much useful information even if it wasn't noise. This part of the algorithm returns the bounding box coordinates, as well as the centroid of the detected object.

### Particle Tracking
The tracking algorithm is based on Kalman filters and the Hungarian Algorithm. Kalman filters are useful in incorporating multiple (error prone) measurements into a physical model to predict an object's state in the future. The Hungarian Algorithm finds an optimal assignment of one set of objects to another, based on the cost of assigning any particular pair. Each unique particle will be assigned a unique particle ID, which will simply be a monotonically increasing integer starting at 0.

The general idea is as follows: For a given frame, there will be a number of particles detected that correspond to the particles in the previous frame, though at slightly different positions.  There is a one-to-one correspondence between particles in the current frame and the previous frame, and we simply need to decide which particles correspond to each other. We can predict where the previous frame's particles will move to using the Kalman filter, and then we can calculate the cost of assigning every particle in the current frame to every particle in the previous frame. Then we can apply the Hungarian Algorithm to find the mapping which minimizes the overall cost. 

The Kalman filter has a state space of 8 dimensions; it measures the following particle parameters: 
* x
* y
* x velocity
* y velocity
* bounding box height
* bounding box width
* bounding box height "velocity" - essentially change in height per frame
* bounding box width velocity

However, there are some complications with this. For example, when a particle appears or disappears, we have to actually assign a dummy column or row to ensure the cost matrix is square. Even worse however, is that even if both frames have n particles, we don't know if all n particles were mapped to the next frame, or if a particle actually disappeared, and an entirely new particle appeared.  The way we handle this is to first assume that no pairs of particles simultaneously appear/disappear. We apply the Hungarian algorithm to find the best assignment, then we iteratively add pairs of appearing/disappear particles to the matrix, calculating the optimum assignment for each, until either the total cost is greater than the previous iteration, or all of the assignment costs fall below a certain "low cost" threshold.

Furthermore, we need to consider the causes of a particle "disappearing"; it could truly disappear, or it could have simply moved behind another larger particle only to reappear later, or maybe the particle detection algorithm failed to detect it for a frame if it was too small. Our solution to this is to have an occlusion counter which tracks for how many frames the particle is occluded. If the particle is not detected for 3 frames, it is deemed to have truly disappeared. When an particle is occluded, its Kalman filter is updated with the prediction itself as the measurement, which is a crude way to maintain the "correct" prediction for the next frame.

The cost function is composed of several metrics applied in equal weight. First, the distance between the prediction of the previous particle and the current particle is considered. Especially with a high frame rate, particles are not likely to move very much between frames, with the exception of microexplosions, so we can consider large distances between a particle in 2 frames to indicate different identities.

A similar argument can be made for the size of the bounding box; as it is not likely to change very much quickly, it can also be used to identify particles. We also consider the brightness of the particles at their centroid, which is also generally constant. Finally, there is a cost penalty for matching occluded particles. The idea here is that we want to make it increasingly costly to bring a occluded particle out of occlusion the longer it has been occluded.  

### Calibration
Before we can extract actual spectral data and curve fit it against temperature, we must be able to translate raw data from the camera into spectral data. Specifically, the camera produces data in units of "pixel number" and pixel brightness, and this needs to be converted to wavelength and spectral radiance respectively. Furthermore, the spectral lines appear on screen a certain translation and rotation away from the source particle; this transformation needs to be captured as well. Finally, the camera itself and the physical optics setup each have their own frequency response which will affect the data, so this needs to be calibrated out. The calibration data we use can be found in the `calibrationData` folder. 

First, we look at an image of a broadband lamp that has had its spectra lines passed through a physical filter to only pass known wavelengths. By recording on what pixel values known spectral lines (450 nm, 532 nm, 633 nm) fall, we can make a linear interpolation and determine the pixel-wavelength scaling factor (the relation is assumed to be linear). 

Then, we look at essentially the same image, but with the physical filter removed so that the full spectra is shown. We can also measure the rotation with respect to the image frame and correct for it by rotating back. We can now determine the translation to map the particle to the spectra line (specifically its centroid). After subtracting the background, we measure the spectral line in terms of pixels in the same way we would measure the real data, and then divide by the known spectra of the broadband lamp, retrieved from the manufacturer's website. 


### Raw Spectra Extraction
After finding the bounding boxes of each particle in a given frame, we analyze the rest of the image to extract spectral data. 

In the simplest case, a spectral line is not overlapped by any other line, and we can simply "read it off". We apply the translation to the original particle's coordinates, and then, starting from a known distance to the left of the mapping, scan to a known distance to the right of the centroid. The height over which we scan is the same height as the original particle's bounding box. We then average over height and width of the particle's bounding box. It seems to be that spectral lines are vaguely diamond in shape, in that they taper towards the ends. Since our averaging region is rectangular, this means that, towards the ends, more background color will be averaged into the spectra. However, this was also the case for the calibration process, and so it is divided away in the calibration. 

Unfortunately, this simplest case is uncommon because of overlaps with other spectral lines. The next case is a "partial conflict"; this is when the spectral line of interest is partially overlapped vertically, but still maintains a horizontal band free of intersection. 
It would appear like this:
```
    ------------------------------------------------
    |                                              |
----|-----------------------------------------     |
|   -----------------------------------------|------
|                                            |
----------------------------------------------	
```
In this case, both spectra regions have a horizontal band that is free from overlap, and an average spectra can be taken for either line over the intersection-free zone. Of course, there can be other spectra lines that intersect in other places, but as long as there is a band of pixels at least 2 pixels tall (I exclude bands of height 1 for having too much noise), the spectra can still be extracted. 

If there are still unknown spectra yet due to overlap, I then try to subtract out already extracted spectra from overlaps. The image taken from the camera is essentially linear, in that you can add or subtract individual spectral lines to produce the final image. Therefore, if we can subtract known spectra away from the regions of overlap, we can iteratively reduce the number of conflicts, hopefully allowing us to determine spectra by the above methods.

If there are still conflicts remaining, then we must apply a different technique to resolve intersections. We know that the spectral measurements are modeled after Planck's Law passed through the camera's response. This means that if there are n particles participating in a conflict, we can solve an optimization problem involving n independent temperatures to try to reproduce the original measurement as best as possible. We know what the physical offsets between the actual particles, and this corresponds to physical offset between the spectral lines, which we incorporate into the optimization problem. Once we recover the temperatures of the particles, we can determine the relative "weight" of each theoretical individual spectra compared to the theoretical sum, and use that proportion of the original measurement, since again, everything is linear. 
After the extraction is done, we record the particles temperature, converted spectra, and "spectra code" which simply describes the method used to find it:
* 0: No conflicts with the original spectral line
* 1: Partial conflict(s), but able to find isolated strip
* 2: No conflicts after subtracting out previously known spectra
* 3: Partial conflict(s), but able to find isolated strip after subtracting out previously known spectra
* 4: Used optimization on temperature parameters (see below)

### Conflict Resolution and Temperature Extraction

Once we have a single spectra, or a measurement containing multiple spectra, we need to extract temperature from it. We can consider the single spectra use a trivial case of a conflict case, a conflict with only 1 source. Initially, we used a Least Squares regression from `scipy` that worked reasonably well. 

However, we also tested a brute force algorithm that searched exhaustively through the temperature space. Unfortunately, this process takes exponential time in the number of particles, so it would be much too slow to always use. However, we added some modifications to improve it's performance. We assume that there are not too many local minimums in the cost function (which is simply the sum of the absolute difference of the measurement against a proposed solution). For each temperature dimension, we segment it into 12 points and run brute force on it. Then, around the point with the lowest cost, we run brute force again, in a much smaller range around it, again, segmenting this small range into 12 points. We do this iteration 3 times to (hopefully) converge reasonably close to the global minimum. From external physical knowledge, we also assume that the majority of the temperatures we will be seeing fall in the range from 2000 to 3000K, and having a narrower range helps brute force search more exhaustively. In addition, there are only 1000 temperature curves corresponding to the (integer) temperatures in the range [2000, 3000], so we cache these and save some time from recalculating them.

For our final method, if the number of particles in the conflict is less than 5, we run both Least Squares and brute force, and choose the solution with the lower cost, which is almost always brute force. For conflicts with 5 or more particles, we only run least squares.  

### Data Storage
All the data we extract is ultimately stored in particle objects, in the `particle.py` file. This includes all particle information (bounding box, brightness, occlusion count), and spectra information (spectra, temperature, spectra code), as well as particle ID, the frame number it first appeared in, and its Kalman filter. 

The particles are indirectly accessed through `database.py`, which, among other responsibilities, maintains a dictionary of particles, with particle ID as the key. When all the processing is over, `mainExtractData.py` will simply pickle the dictionary as it is, containing all the particle objects. This makes it easy for other "main" files, such as `mainCreateCompiledVideo.py` or `mainCreateCSV.py` to read in the data in a usefully structured manner for their own processing. 

### "Main" Files
In order to use this entire repository, you only really need to interact with the "main" files. The first place to start is `mainExtractData.py`, which will create the pickle file that contains all the extracted data.  As mentioned in the Quick Start Usage, `mainCreateCSV.py` creates a CSV containing all useful information from the pickle file, and `mainCreateCompiledVideo.py` renders a video that incorporates a lot of the data. In the video, I color the particles based on their color temperature, but since the range of temperatures is only 2000 to 3000K, this does not provide very good visual contrast, so I linearly map the actual temperature to the range 1500 to 3500K, just to have a little more contrast (but there still isn't that much). `mainCreateSpectraVideo.py` creates a video focusing on the spectral information for each particle, though it is unfortunately rather slow to render compared to the others.`mainDebug.py` is simply a development/debugging tool that steps through each frame sequentially, and it could be of use to others. Users should also feel free to modify any of the main files to suit their purposes, or any file at all. 

### Limitations
At every step in this software, error and noise are unfortunately introduced. The particle detection technique tends to miss extremely small particles, as well as particles that are close enough together that, according to the thresholding, they are seen as only 1 particle. The particle tracking algorithm is similarily touchy, and can often confuse two particles. Tracking works best with large, slow moving particles that are fairly isolated from other particles. If you are interested in focusing on a particular particle, it would be best to look at the sort of video produced by `mainCreateCompiledVideo.py` to visually identify particles and track any changes to its particle ID. The spectral extraction sometimes produces spectra that are clearly incorrect, and these should be ignored, as they come from spectral lines with a lot of overlap. There certinly might be a bug in the spectra extraction, I have not throughly tested it. In addition, the videos I have looked at tend to have this white speckle noise, and if a speckle shows up on the spectra, it will cause a sharp spike in the spectra; this should also be ignored. 


### Acknowledgements
I would like to thank Professor Mark Foster and Dr. Milad Alemohammad for their guidance and assistance throughout the project, Velat Kilic for occasional help, and Petr Klus for a color temperature conversion function
