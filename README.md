# Homography_Mixtures
## Implement 

[[Calibration-free rolling shutter removal](https://smartech.gatech.edu/bitstream/handle/1853/48740/2012-Grundmann-CRSR.pdf?sequence=1&isAllowed=y)]

## Requirements

```
pip3 install -r requirements.txt
```
## Compile CPP
```
cd SGridSearch
cmake .
make
```
## Demo
```
python3 find_homography.py --img1 ois_off_frame-479.jpg --img2 ois_off_frame-480.jpg
```
