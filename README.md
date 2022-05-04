# pytoshearlab

pytoShearLab is a toolbox containing a 2D shearlet transform in pytorch, based on [pyShearLab](http://na.math.uni-goettingen.de/pyshearlab/) by Stefan Look and ShearLab 3D by Rafael Reisenhofer.

## Implementation Details

The toolbox currently consists of a 2D shearlet transform for $N\times N$ size images, where shearlet transform together with inverse and adjoint operations are available.

## Repository Structure
The repository is structured at follows:

* the base directory contains the main implementation [`pytoShearLab2D.py`](pytoShearLab2D.py) and a test script [`test.py`](test.py)
* 'utils' contains filter and fourier transform tools 
* 'data' contains an exemplary image for testing

## Dependencies

* numpy
* pytorch
* matplotlib (to run [`test.py`](test.py))
* cv2 (to run [`test.py`](test.py))

pytoShearLab was tested with python 3.7.7 and pytorch 3.7.1. on Linux (Ubuntu 18.04).

## Usage

After installing all of the required dependencies above you can test pytoShearLab using

~~~
python test.py
~~~
