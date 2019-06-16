# Simple graphics library

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

# About library
SKGL is a simple graphics library implementation (my opengl implementation) that render on real-time 3D objects in OBJ file format. This is not famework or graphics engine, this is self implementation of open gl. 

# Hardware render
This library used CUDA SDK for render 3D objects on graphics accelerator.

# Result render
![rendered on library image](https://raw.githubusercontent.com/SlavickKuzmin/SKGL/master/ReadMeResources/ResultRenderInRealTime.png)

Example on YouTube: https://youtu.be/E1w57muC8z0

Library based on github project: https://github.com/ssloy/tinyrenderer
For UI used nuklear :https://github.com/vurtun/nuklear
For change screen buffers used SDL 2.0, because operation system deny write directly to video memory.

P.S. Is my diploma work for KPI, Kiev 2019. Name SKGL is abbriviation of Slavick Kuzmin Graphics Library.
