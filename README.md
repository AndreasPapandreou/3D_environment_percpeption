# README #

## Intro ##

* Quick summary

## Pipeline ##
The pipeline has been split into two stages as you can observe from the next figures:

**Part A**

![alt text](https://github.com/AndreasPapandreou/3D_environment_percpeption/blob/master/res/pipeline_parta.png?raw=true)

**Part B**

![alt text](https://github.com/AndreasPapandreou/3D_environment_percpeption/blob/master/res/pipeline_partb.png?raw=true)


## How do I get set up? ##

The project runs in linux distribution using the CLion IDE, so the next steps are referred to any linux system.

1. Building **GLFW** from their webpage's download page https://www.glfw.org/download.html. Select the source package and
   run the below steps :
    - cd glfw-3.3
    - cmake .
    - make
    - make install
2. Setting up **GLAD**. Go to the web service https://glad.dav1d.de/, make sure the language is set to C++ and in the API
   section, select an OpenGL version of at least 3.3 (which is what we'll be using for these tutorials; higher versions
   are fine as well). Also make sure the profile is set to Core and that the Generate a loader option is ticked.
   Ignore the extensions (for now) and click Generate to produce the resulting library files. Copy both include folders
   (glad and KHR) into your include directory (or add an extra item pointing to these folders), and add the glad.c file
   to your project.
3. Build nlohmann/json, which is a C++ library that allows manipulating JSON values.
    Link : https://github.com/nlohmann/json
4. Add MathGeoLib from https://github.com/juj/MathGeoLib/tree/master/src.
5. Add pcl from http://www.pointclouds.org/documentation/tutorials/compiling_pcl_posix.php.
6. Add eigen from https://dritchie.github.io/csci2240/assignments/eigen_tutorial.pdf.

## User inputs ##

1. User must define two inputs in main. The first argument should be the path to binary lidar data and the second one
must be the path to classifier model.
2. Data will be uploaded in the future

   ### Some explanations about ranger
1. How to train classifier :
   ./ranger --verbose --file path/to/file.dat --depvarname type --treetype 1 --ntree 1000 --nthreads 8 --write ranger_out.forest

## Results ##
