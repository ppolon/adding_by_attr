MATLAB code for Adding Samples by Learned Attributes

 Version: 0.2

 Publication:
 Adding Unlabeled Samples to Categories by Learned Attributes
  by Jonghyun Choi and Mohammad Rastegari and Ali Farhadi and Larry S. Davis
  In Proceedings of IEEE CVPR 2013
 
 Code is written by Jonghyun Choi (jhchoi@umiacs.umd.edu)
 Report any bugs to Jonghyun Choi (jhchoi@umiacs.umd.edu)
 
 Copyright (c) 2013 Jonghyun Choi

 License: The MIT License
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 Required Software:
   1. Matlab Package of LibLinear: http://www.csie.ntu.edu.tw/~cjlin/liblinear/#download

 Recommended Software:
   1. Matlab Package of VL_FEAT: http://www.vlfeat.org/download.html
   2. Matlab Package of Attribute Discovery: http://www.umiacs.umd.edu/~mrastega/paper/dbc.zip
   
 NOTE: Please install the above software and add the path to the MATLAB paths.
 NOTE: Please unextract the data file to "./data" in the directory of code
 
 Files:
    demo.m              : demo script
    addByGnE.m          : main function of one iteration of Eq.(1)
    precisionRecall.m   : compute precision and recall curve from PASCAL evaluation code package
    ./dbc               : http://www.umiacs.umd.edu/~mrastega/paper/dbc.zip
    train.mexw64        : LibLinear binary compiled for 64 bit Windows
    vl_homkermap.mexw64 : homogeneous kernel mapping in VL_FEAT compiled for 64 Windows

 Usage: type "help addByGnE" or Please refer to demo script (demo.m)
 
 Demo: Please type the following in the Matlab prompt
  >> demo
