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
## Citation
Please cite the related paper if you find our code useful!
```
@inproceedings{grundmann2012calibration,
  title={Calibration-free rolling shutter removal},
  author={Grundmann, Matthias and Kwatra, Vivek and Castro, Daniel and Essa, Irfan},
  booktitle={2012 IEEE international conference on computational photography (ICCP)},
  pages={1--8},
  year={2012},
  organization={IEEE}
}

@article{liu2021deepois,
  title={DeepOIS: Gyroscope-guided deep optical image stabilizer compensation},
  author={Liu, Shuaicheng and Li, Haipeng and Wang, Zhengning and Wang, Jue and Zhu, Shuyuan and Zeng, Bing},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={32},
  number={5},
  pages={2856--2867},
  year={2021},
  publisher={IEEE}
}
```
