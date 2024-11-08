function [x, li, L] = attn_ml_sps_OT(yt, bt, G, Gopt, x, rt, maxit, li, aa)
% An nerual optimzation transfer is used to solve the attenuation map
% reconstruction by using following two steps:
%
%    1. Attenuation map reconstruction from PET emssion data
%    2. neural-network learning in image domain based on weighted MSE loss
%    derived from optimzation transfer.
%
% sqlli@ucdavis.edu 01-22-2022

%% check inputs

numprj = prod(Gopt.prjsiz);
numfrm = size(yt,2);
numbin = size(yt,1)/numprj;

if nargin<2 | isempty(bt)
    bt = ones(numprj,numfrm);
end
if nargin<6 | isempty(rt)
    rt = ones(numprj*numbin,numfrm);
    yt = yt + rt;
end
if nargin<8
    li = [];
end
if nargin<9 
    aa = [];
end
if isempty(aa)
    aa = proj_forw(G, Gopt, ones(size(x)));
end
%% iterate
for it = 1:maxit
    
    % foward projection
    if isempty(li)
        li = proj_forw(G, Gopt, x);
    end
    
    % gradient
    lt = repmat(li,[1 numbin]);
    lt = repmat(lt(:),[1 numfrm]);
    yb = bt.*exp(-lt) + rt;
    yr = (1-yt./yb).*(yb-rt);
    yr = reshape(yr, [numprj numbin numfrm]);
    hi = sum(sum(yr,3),2);
    gx = proj_back(G, Gopt, hi);
    
    % optimal curvature and the optimization transfer weight
    nt = sum(trl_curvature(yt, bt, rt, lt, 'oc'),2);
    nt = reshape(nt,[numprj numbin]);
    wx = proj_back(G, Gopt, sum(nt,2).*aa, numfrm);
    
    % objective function
    L(it) = sum(yt(:).*log(yb(:)) - yb(:));
    
    % update image
    x(Gopt.mask) = x(Gopt.mask) + gx(Gopt.mask) ./ wx(Gopt.mask);
    x(Gopt.mask(:)&wx(:)==0) = 0;
    x = max(0,x);
    
    % neural network learning based on weighted MSE lose
    temp = x;
    weight_img = wx;
    temp = reshape(temp,Gopt.imgsiz);
    weight_img = reshape(weight_img, Gopt.imgsiz);
    u_int = temp(Gopt.trunc_range{1}, Gopt.trunc_range{2});
    weight_int = weight_img(Gopt.trunc_range{1}, Gopt.trunc_range{2});
    dump(u_int,sprintf('inter_img.img'));
    dump(weight_int,sprintf('weight.img'));
    system('python DIP_step.py');
    result = touch('DIP_output.img');
    result = reshape(result,Gopt.imgsiz_trunc);
    DIP_out = zeros(Gopt.imgsiz);
    DIP_out(Gopt.trunc_range{1}, Gopt.trunc_range{2}) = result;
    DIP_out = DIP_out(:);
    if DIP_out == 0
       x = x;
    else
       x = DIP_out;
       x = x(:);
    end
    
    % update projection
    if it<maxit
        li = proj_forw(G, Gopt, x, numfrm);
    end
    
end
