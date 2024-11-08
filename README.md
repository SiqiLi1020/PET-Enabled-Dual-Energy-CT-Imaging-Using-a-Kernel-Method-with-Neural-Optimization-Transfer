# Single-Subject Deep-Learning Image Reconstruction with a Neural Optimization Transfer Algorithm for PET-Enabled Dual-Energy CT Imaging

This demo code shows how to reconstruct the gamma-ray CT (gCT) image from PET emission data. The reconstructed high-energy (511keV) gCT is then combined with the already-existing X-ray CT image to form dual-energy pairs. 

The proposed method, called "Neural KAA", is described in:

    S. Li, Y. Zhu, B. A. Spencer, and G. B. Wang, "Single-Subject Deep-Learning Image Reconstruction With a Neural Optimization Transfer Algorithm for PET-Enabled Dual-Energy CT Imaging," in IEEE Transactions on Image Processing, vol. 33, pp. 4075-4089, 2024.

Program Authors: Siqi Li

Last date: 11/08/2024


# Prerequistites:
	Python 3.7 (or 3.x)
	GPU-based PyTorch
	Matlab

# Overview



The neural KAA reconstruction consists of three separate steps: 

(1) a KEM step for image update from the projection data

(2) a deep-learning step in the image domain for updating the kernel coefficient image

# Neural KEM for dynamic PET reconstruction:

a). You can run 'demo_Nerual_KEM.m' to test the proposed method on Zubal phantom simulation that we used in the paper.

b).	'DIP_step.py' is a function to run the step of deep coefficient prior (deep learning), and the whole reconstruction is implemented on Matlab

# Required packages for PET reconstruction:

a).	To use this package, you need to add the KER_v0.11 package into your matlab path by
  	running setup.m in matlab. KER_v0.11 package can be downloaded from:
   
	https://wanglab.faculty.ucdavis.edu/code

b).	To test the algorithms in the package, run "demo_Neural_KEM.m" in the current folder. You
  	may need your own system matrix G or use Jeff Fessler's IRT matlab toolbox to 
  	generate one. IRT can downloaded from 
  
      	http://web.eecs.umich.edu/~fessler/code/index.html

# License
This package is the proprietary property of The Regents of the University of California.
 
Copyright © 2019 The Regents of the University of California, Davis. 
All Rights Reserved. 
 
This software may be patent pending.
 
The software program and documentation are suppluntitled.mied "as is", without any 
accompanying services from The Regents, for purposes of confidential discussions 
only. The Regents does not warrant that the operation of the program will be 
uninterrupted or error-free. The end-user understands that the program was 
developed for research purposes and is advised not to rely exclusively on 
the program for any reason.
 
IN NO EVENT SHALL THE REGENTS OF THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY
PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, 
INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, 
EVEN IF THE REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE REGENTS 
SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE REGENTS HAS NO 
OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS. 


# Contact
Please feel free to contact me (Siqi Li) if you have any questions: sqlli@ucdavis.edu
