clear, clc, close all, format long

u_exact = double(imread('cameraman.tif'));

umax = max(max(u_exact));
u_exact = u_exact/umax;
N=size(u_exact,1); kernel=Kegen(N,300,10);
nxy =128; nx = nxy; ny = nxy; % Resize to reduce Problem
u_exact=imresize(u_exact,[nx nx]); kernel=imresize(kernel,[nx nx]);     
hx = 1 / nx; hy = 1 / ny; N=nx; hx2 = hx^2;
%  Extend kernel and compute its 2-d Fourier transform. Then use this to 
%  compute K'*z and kstark_hat, the 2-d Fourier transform for K'*K. 
kernel=kernel/sum(kernel(:));
m2 = 2*nx; nd2 = nx / 2; kernele = zeros(m2, m2) ;
kernele(nd2+1:nx+nd2,nd2+1:nx+nd2) = kernel ; %extention kernel
k_hat = fft2(fftshift(kernele)) ; clear kernele
z = integral(u_exact,k_hat,nx,nx);  % Blur Only  PLUS NOISE if needed
%Parameters
beta =0.001;  alpha = 1e-8;  n = nx^2;  m = 2*nx*(nx-1);  nm = n + m;

[B] = computeB(nx);
u0 = zeros(nx,nx);  

tol = 1e-8; maxit = 1000; 
u = z;
fprintf('k     c_k        psnr      Negitives  \n')
fprintf('-     ----       -----     -------- \n')

 figure;   imagesc(u_exact);colormap(gray);
 s=sprintf('exact image');s=title(s);  

 zpsnr = psnr(z,u_exact);
 fprintf('%d      %d      %11.9g    %d \n',0,0,zpsnr,sum(sum(z<0)))
 figure;  imagesc(z);ss=sprintf('blured image psnr = %0.5g',zpsnr);ss=title(ss);   colormap(gray);

b0 = integral(z,conj(k_hat),nx,nx); b0 = b0(:);
% ------- Solve the case when k = 0  ---------
k = 1;
c = 0.0000095;  d = 5;
p = 2;
xeps = 0.005;
lam = 0 * hx2 * z;
wgh = 1.1;
w = max( 0 , u - lam/c - xeps );
b = b0 + wgh*hx2*lam(:) + hx2 * c * ( w(:) - xeps );


[D,C,A] = ComputeDCA(u,nx,m,beta);
BD = B'*inv(D) ;
L1 = BD*B;
L2 = BD*C*inv(D)*B;
L3 = A*BD*B;
L = (L1)^2 + L2 + L3;
    
[U,flag,rr,iter,rv]= gmres(@(x)AKKL(nx,x,k_hat,L,alpha,c),b,[],tol,maxit);
  
u = reshape(U,nx,nx); Upsnr = psnr(u,u_exact);
figure;  imagesc(u);colormap(gray);
ss=sprintf('k = %d , psnr = %0.5g',k,Upsnr);ss=title(ss);colormap(gray);
fprintf('%d     %3.1g    %11.9g   %d \n',k,c,Upsnr,sum(sum(U<0)))

% ------- End Solve the case when k = 0  ---------
for k=2:10
    lam = lam + c * max( -u + xeps , -lam/c );
    c = d * c ; % d in [4, 10 ]
    w = max( 0 , u - lam/c - xeps );
    b = b0 + wgh* hx2 *lam(:) + hx2 * c * ( w(:) - xeps );

[D,C,A] = ComputeDCA(u,nx,m,beta);
BD = B'*inv(D) ;
L1 = BD*B;
L2 = BD*C*inv(D)*B;
L3 = A*BD*B;
L = (L1)^2 + L2;
    
[U,flag,rr,iter,rv]= gmres(@(x)AKKL(nx,x,k_hat,L,alpha,c),b,[],tol,maxit);
  
    u = reshape(U,nx,nx); Upsnr = psnr(u,u_exact);
    figure;  imagesc(u);colormap(gray);
    ss=sprintf('k = %d , psnr = %0.5g',k,Upsnr);ss=title(ss);colormap(gray);
    negative_pixels = sum(sum(U<0));
    fprintf('%d     %3.1e  %11.9g   %d \n',k,c,Upsnr,negative_pixels)
    if negative_pixels == 0; break; end
end
%---------------------------------------------------------------
%% ke_gen.m:    K = ke_gen(n, tau, radius);
%%     Generates a n^2 x n^2 kernel PSF (point spread function)
%%     radius=4 and tau=200  (default)
function K = Kegen(n, tau, radi);
if nargin<1,help Kegen;return; end
if nargin<2, tau=200; end
if nargin<3, radi=4; end
K=zeros(n);
R=n/2; h=1/n; h2=h^2;
%RR=n^2/radi+1; 
RR=radi^2;

if radi>0 
%___________________________________________

for j=1:n
  for k=1:n
    v=(j-R)^2+(k-R)^2;
    if v <= RR,
      K(j,k)=exp(-v/4/tau^2);
    end;
  end;
end;
sw=sum(K(:));
K=K/sw; %*tau/pi;

else radi<0 
%___________________________________________
 range=R-2:R+2;
 K(range,range)=1/25;
end
end
  function Ku = integral(u,k_hat,nux,nuy)
%
%  Ku = integral_op(u,k_hat)
%
%  Use 2-D FFT's to evaluate discrete approximation to the 
%  2-D convolution integral
%
%    Ku(x,y) = \int \int k(x-x',y-y') u(x',y') dx' dy'.
%
%  k_hat is the shifted 2-D discrete Fourier transform of the 2_D 
%  kernel evaluated at node points (x_i,y_j), and then extended.
%  u is also assumed to be evaluated at node points (x_i,y_j).
%  The size of k_hat may be different that of u, due to extension.

  [nkx,nky] = size(k_hat);
  h=1/nkx;
  Ku = real(ifft2( ((fft2(u,nkx,nky)) .* k_hat)));
  if nargin == 4
    Ku = Ku(1:nux,1:nuy);
  end
 %   Ku = Ku*h^2; %% Ke tries Apr 06
  end
  
function [B] = computeB(nx)
  e = ones(nx,1);
E = spdiags([0*e -1*e e], -1:1, nx, nx);
E1 =E(1:nx-1,:);
 
M1=eye(nx,nx);
B1=kron(E1,M1);
 
E2 = eye(nx);
M2 = spdiags([0*e -1*e e], -1:1, nx-1, nx);
B2 = kron(E2,M2);
 
B = [B1;B2];
end

function [D,C,A] = ComputeDCA(U,nx,m,beta)
h0=1/nx;
%[X,Y] = meshgrid(h0/2:h0:1-h0/2);

nn = size(U,1);
UU = sparse(nn+2,nn+2);

% we are using reflection bounday conditions 
% another word, we are using normal boundary condition to be zero
UU(2:nn+1,2:nn+1) = U;
UU(1,:) = UU(2,:);
UU(nn+2,:) = UU(nn+1,:);
UU(:,1) = UU(:,2);
UU(:,nn+2) = UU(:,nn+1);
%------------------ Matrix D ------------------
Uxr = diff(U,1,2)/h0; % x-deriv at red points
xb = h0/2:h0:1-h0/2;   yr=xb;
yb = h0:h0:1-h0;       xr=yb;
[Xb,Yb]=meshgrid(xb,yb);
[Xr,Yr]=meshgrid(xr,yr);
Uxb = interp2(Xr,Yr,Uxr,Xb,Yb,'spline');
 
 Uyb = diff(U,1,1)/h0; % y-deriv at blue points
 Uyr = interp2(Xb,Yb,Uyb,Xr,Yr,'spline');
  
 Dr = sqrt( Uxr.^2 + Uyr.^2 + beta^2 );
 Db = sqrt( Uxb.^2 + Uyb.^2 + beta^2 );
 mm1 = size(Dr,1);
 
 Dvr = Dr(:);  Dvb = Db(:); Dv=[Dvr;Dvb];
 
 ddd = [ sparse(m,1) , Dv , sparse(m,1) ];
 D = spdiags(ddd,[-1 0 1],m,m);
 %-------------------------Matrix C----------------------------
 Wxr = diff(UU,3,2)/h0; % x-deriv at red points
 Wyb = diff(UU,3,1)/h0; % y-deriv at blue points
 
   
Wr = Wxr(1:mm1,:); 
Wb = Wyb(:,1:mm1); 
 
 Dwr = (Wr(:).*Uxr(:))./Dr(:);  Dwb = (Wb(:).*Uyb(:))./Db(:); Dw=[Dwr;Dwb];
 
 www = [ sparse(m,1) , Dw , sparse(m,1) ];
 C = spdiags(www,[-1 0 1],m,m);
 
 %-------------------- Matrix A -----------------------------
 
 E = zeros(nx,nx); 
 E(1,1)=1; E(nx,nx)=1;
 M=speye(nx,nx);

 A1 = kron(M,E);
 A2 = kron(E,M);
 A = 2*(A1 + A2)/(beta*h0);
 %-----------------------------------------------------------
 
end

function [y] = AKKL(nx,x,k_hat,L,alpha,c)
x = reshape(x,nx,nx);
y1 = integral(x,k_hat,nx,nx);
y1 = integral(y1,conj(k_hat),nx,nx);
y = y1 + reshape(alpha * L * x(:),nx,nx);
hx2 = (1/nx)^2;
y = y(:) + c * hx2 * x(:);
end

  function p = psnr(x,y)

% psnr - compute the Peack Signal to Noise Ratio, defined by :
%       PSNR(x,y) = 10*log10( max(max(x),max(y))^2 / |x-y|^2 ).
%
%   p = psnr(x,y);

d = mean( mean( (x(:)-y(:)).^2 ) );
m1 = max( abs(x(:)) );
m2 = max( abs(y(:)) );
m = max(m1,m2);

p = 10*log10( m^2/d );
  end
  