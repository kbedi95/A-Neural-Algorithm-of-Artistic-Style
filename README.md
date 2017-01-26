A-Neural-Algorithm-of-Artistic-Style
======================================

An implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) in Tensorflow.

Dependencies
--------------------
Python 3.5, pip, virtualenv

Installation
-----------------
```shell
virtualenv styleEnv --python=python3.5
source styleEnv/bin/activate
pip install -r requirements.txt
```


Running the Application
--------------
```shell
python neuralStyle.py path/to/content_image path/to/style_image

optional arguments:
  -h, --help            show this help message and exit
  -tp TARGET_PATH, --target_path TARGET_PATH
  -cw CONTENT_WEIGHT, --content_weight CONTENT_WEIGHT
  -sw STYLE_WEIGHT, --style_weight STYLE_WEIGHT
  -tvw TV_WEIGHT, --tv_weight TV_WEIGHT
  -it ITERATIONS, --iterations ITERATIONS
  -f FREQUENCY, --frequency FREQUENCY
```

Sample Results
----------------------
Content: ![](/samples/DwadeStarry/Dwade.png) Style: <img src="/samples/DwadeStarry/starry.jpeg" width="220" height="220">  Generated: ![](/samples/DwadeStarry/DwadeStarry.png)
<br><br><br><br>
Content: ![](/samples/BradPicasso/Brad.png) Style: <img src="/samples/BradPicasso/Picasso.jpeg" width="220" height="220"> Generated: ![](/samples/BradPicasso/BradPicasso.png)
<br><br><br><br>
Content: ![](/samples/TorontoTrees/Toronto.png) Style: <img src="/samples/TorontoTrees/Trees.jpg" width="220" height="220"> Generated: ![](/samples/TorontoTrees/TorontoTrees.png)
<br><br><br><br>
Content: ![](/samples/StreetFace/Street.png)  Style: <img src="/samples/StreetFace/Face.jpeg" width="220" height="220"> Generated: ![](/samples/StreetFace/StreetFace.png)


