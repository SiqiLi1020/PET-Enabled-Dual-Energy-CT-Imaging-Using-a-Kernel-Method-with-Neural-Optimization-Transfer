# Single-Subject Deep-Learning Image Reconstruction with a Neural Optimization Transfer Algorithm for PET-Enabled Dual-Energy CT Imaging

This demo code shows how to reconstruct the gamma-ray CT (gCT) image from PET emission data. The reconstructed high-energy (511keV) gCT is then combined with the already-existing X-ray CT image to form dual-energy pairs. 

The proposed method, called "Neural KAA", is described in:

    S. Li, Y. Zhu, B. A. Spencer, and G. B. Wang, "Single-Subject Deep-Learning Image Reconstruction With a Neural Optimization Transfer Algorithm for PET-Enabled Dual-Energy CT Imaging," in IEEE Transactions on Image Processing, vol. 33, pp. 4075-4089, 2024.

Program Authors: Siqi Li and Guobao Wang

Last date: 11/08/2024


# Prerequistites:
	Python 3.7 (or 3.x)
	GPU-based PyTorch
	Matlab

# Overview

![figure](https://github.com/user-attachments/assets/b9a35c03-a26d-4fbc-925f-cceb4be4005d)

The neural KAA reconstruction consists of three separate steps: 

(1) PET activity image update (MLEM) 

(2) gCT image update (KTR)

(3) A weighted least-square neural-network learning in the gCT image domain

# Neural KEM for dynamic PET reconstruction:

a). Please download the system matrix folder from https://drive.google.com/drive/folders/1rx_CsvE8D7N2HLML14VUsjNmG2sssSeK?usp=sharing, and put them into folder "/data/GE690TOF2D" (need to create)

b). Then run 'Test_demo.m' to test the proposed method and compare it with other methods in the simulation study that we used in the paper.

c). 'DIP_step.py' is a function running the step of a weighted least-square neural-network learning in the gCT image domain, and the whole reconstruction is implemented on Matlab.

# Required packages for PET reconstruction:

To run the demo code, you need to add the "KER_v0.11" and "PLOT_v1.0" packages into your Matlab path by running setup.m in Matlab. Both packages can be downloaded from:

	https://wanglab.faculty.ucdavis.edu/code

# Future effort:

Please refer to our latest published paper on EJNMMI regarding the implementation of real data. We hope the implementation of PET-enabled DECT on real data will come soon!

	Zhu Y, Li S, Xie Z, Leung EK, Bayerlein R, Omidvari N, Cherry SR, Qi J, Badawi RD, Spencer BA, Wang GB. Feasibility of PET-enabled dual-energy CT imaging: First physical phantom and initial patient study results. European Journal of Nuclear Medicine and Molecular Imaging, accepted, November 2024. arXiv:2402.02091. 3 Feb 2024. https://doi.org/10.48550/arXiv.2402.02091.

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
