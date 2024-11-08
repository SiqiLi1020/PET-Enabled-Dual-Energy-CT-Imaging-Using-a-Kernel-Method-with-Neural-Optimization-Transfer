% This is a demo to test different image reconstruction algorithms for PET-eanbeld dual-energy CT on simulation data
% NOTE: Please do not distribute this package without permission from the authors

% The related papers are here:

% Standard MLAA:               A. Rezaei, M. Defrise, G. Bal, C. Michel, M. Conti, C. Watson, and J. Nuyts, "Simultaneous reconstruction of activity and attenuation in time-of-flight PET," 
%                              IEEE Trans. Med. Imag., vol. 31, no. 12, pp. 2224-2233, Dec. 2012.

% Kernel MLAA (KAA):           G.B. Wang, "PET-enabled dual-energy CT: image reconstruction and a proof-of-concept computer simulation study," 
%                              Phys. Med. Biol., vol. 65, pp. 245028, Nov. 2020.

% DIP method and neural KAA:   S. Li, Y. Zhu, B. A. Spencer, and G.B, Wang, "S. Li, Y. Zhu, B. A. Spencer and G. Wang, "Single-Subject Deep-Learning Image Reconstruction With a Neural Optimization Transfer Algorithm for PET-Enabled Dual-Energy CT Imaging," 
%                              IEEE Transactions on Image Processing, vol. 33, pp. 4075-4089, 2024.


% Programmer:    Siqi Li, Yansong Zhu and Guobao Wang, UC DAVIS.
% Contact:       sqlli@ucdavis.edu; yszhu@ucdavis.edu; gbwang@ucdavis.edu
% Last update:   10/25/2024

clc;
clear;

% choose the first noisy realization
i = 1;

% count level
count = 5e6;

% choose rectype 'MLAA','KAA','DIP','Neural KAA';
rectype = 'Neural KAA';

% initialization method
initype = 'CT';

run('KER_v0.11/setup');  
run('PLOT_v1.0/setup');
addpath('MLAA functions/');

%% load data

% load the image data at 511 keV
load('data/psct_xcat');

% load the correction data
load(sprintf('data/proj%d',0));

% system matrix - attenuation
load('data/GE690TOF2D/GE690_attn_sysmat_180x3.27mm.mat');
Gopt.imgsiz = [180 180];
Gopt.prjsiz = [281 288];
Gopt.imgsiz_trunc = [128,128]; % to fit for the U-net input
Gopt.trunc_range = {27:154, 27:154}; % truncate from 180 x 180 to 128 x 128.

% system matrix - emission
load('data/GE690TOF2D/GE690_tof2d_sysmat_11tbin_180x3.27mm.mat');
Popt.imgsiz = [180 180];
numbin      = 11;
Popt.prjsiz = [281 288 numbin];

% joint
Gopt.attnFlag = 1;
Popt.emisFlag = 1;
Popt.disp     = 1;
Popt.savestep = 100;
maxit = 600; % 600 is a reasonbale choice

% mask 
Gopt.mask = mask; 

% GCT initial
imgsiz = size(mask);
uinit = CT2LAC(CT,'120','bilinear')/10;
inilabel = '';
if strcmp(initype,'uniform')
    uinit = zeros(imgsiz);
    uinit(mask) = 0.01;
    inilabel = '-UI';
end

% PET activity initial
xinit = zeros(size(mask));
xinit(Gopt.mask) = 1;

% fold
mkdir('result/', sprintf('xcat_proj%dm_rec',count/1e6));

% load noisy projection
load(sprintf('data/proj%d', i));

% %PET initial
if ~strcmp(rectype,'mlem')
    load(sprintf('data/%s_%d','mlem', i));
    xinit = out.xest(:,end);
end


%% Build kernel matrix

% standard kernel matrix
imgsiz_CT = size(CT);
R = buildNbhd(imgsiz_CT, 'clique', 1); % Extract features using a 3x3 patch 
I = [[1:prod(imgsiz_CT)]' R.N]; 
F = CT(I);
F = F * diag(1./std(F,1)); % normalization
sigma = 1;
[N, W] = buildKernel(imgsiz, 'knn', 50, F, 'radial', sigma, 1);
K = buildSparseK(N, W);
Gopt.kernel = K;

%% prior image as the neural network input 
dump(CT(Gopt.trunc_range{1}, Gopt.trunc_range{2}),sprintf('Prior_CT.img'));

%% sub-iteration number setting
sub_iteration = 151;
save('sub_iteration.mat',"sub_iteration");

%% Synergistic reconstruction
switch rectype

    case 'MLAA'
        [u, x, out] = psct_kmlaa(yi, ni, G, Gopt, uinit(:), P, Popt, xinit(:), ri, maxit);
        save(sprintf('result/xcat_proj%dm_rec/%s_%d', count/1e6, 'MLAA', i), 'out');
        
    case 'KAA'
        [u, x, out] = psct_kmlaa(yi, ni, G, Gopt, uinit(:), P, Popt, xinit(:), ri, maxit, K);
        out.kest = K * out.uest;
        save(sprintf('result/xcat_proj%dm_rec/%s_%d', count/1e6, 'KAA', i), 'out');

    case 'DIP'
        [u, x, out] = psct_kmlaa_OT(yi, ni, G, Gopt, uinit(:), P, Popt, xinit(:), ri, maxit, [], sub_iteration);
        save(sprintf('result/xcat_proj%dm_rec/%s_%d',count/1e6, 'DIP', i), 'out');
        
    case 'Neural KAA'
        [u, x, out] = psct_kmlaa_OT(yi, ni, G, Gopt, uinit(:), P, Popt, xinit(:), ri, maxit, K, sub_iteration);
        out.kest = K * out.uest;
        save(sprintf('result/xcat_proj%dm_rec/%s_%d',count/1e6, 'Neural KAA', i), 'out');

end