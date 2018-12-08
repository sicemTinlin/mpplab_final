
function [west,Aest,erro]=gpu_esprit_v6(y,M,Ma,myGPU)
% input: vector y (complex in general)
%        M : order (max. number of complex exponentials expected)
%            M<N/2!
% y is supposed to be composed of terms given by {Aest*exp(j*west*n)}
%                                                 n=1:N
% output: west: complex in general 
%         Aest: complex in general
%         erro: is the mean squared error in energy obtained
% NOTE: west*n = w*d so w = west*(n/d)

  j = 1i;
  y=y(:);          %make sure y is a row vector
  N=length(y);     %get the number of elements             
  i=(1:N)';        %get a column vector of equal length to i
 
  %check GPU memory availability
  n_bits_single_complex = 64;
  n_bits_single_real = 32;
  
  n_bits_needed = length(y)*n_bits_single_complex + n_bits_single_complex * length(y)^2;
  if myGPU.AvailableMemory > n_bits_needed
      y_d = gpuArray(single(complex(y)));
      x_d=hankel(complex(y_d));     %make a hankel matrix out of the data
      x = gather(complex(x_d));
  else
      x = hankel(y);
  end
  
  x=flipud(x(1:M,1:(N-M+1))); %Take the hankel matrix of the data out to row M and column M+1 and discard the rest
  Rx=x*x'/(N-M+1); %find the sample covariance matrix of the new data
  
  
  % check GPU memory availability
  n_bits_needed = length(Rx)^2 * n_bits_single_complex + length(Rx) * n_bits_single_complex;
  if myGPU.AvailableMemory > n_bits_needed
      Rx_d = complex(single(Rx));
      D_d = eig(complex(Rx_d));
      D_d = real(D_d);
      D = gather(D_d);
  else %run on cpu
      D = eig(Rx);
      D = real(D);
  end
  
  [~,ind]=min(D); %find the minimum real eigenvalue of Rx
  thr=D(ind);      % store that minimum eigenvalue of Rx
 
  Rx=Rx-thr*eye(M); % find ASA' (see complex cisoids in noise paper eq.14)
  
  %check that gpu has enough memory free
  n_bits_needed = length(Rx) * n_bits_single_complex + 2 * (length(Rx)^2 * n_bits_single_complex);
  if myGPU.AvailableMemory > n_bits_needed
      Rx_d = gpuArray(single(complex(Rx)));
      [V_d,D_d] = eig(complex(Rx_d));
      [V,D] = gather(complex(V_d),complex(D_d));
  else
      [V,D] = eig(Rx);
  end
  
  
  D=real(diag(D));  % takes the eigenvalues from diagonal matrix D
  Ds=sort(D);       % sort eigenvalues of Cxx
  thr=abs(Ds(M-Ma)); 
 ind=find(D>thr);    %get all eigenvalues greater than that eigenvalue chosen above
  d=length(ind);    % get how many eigenvalues there are that are greater than the eigenvalue chosen above
  Es=V(:,ind);      % get the first d column vectors of V (which are the first d eigenvectors of Cxx)
  if (d==M),        % d must not equal M
    disp('error: d=M!');
  end;
  Es1=Es(1:M-1,:);  % submatrix of eigenvector matrix
  Es2=Es(2:M,:);    % submatrix of eigenvector matrix
  TpT=pinv(Es1)*Es2; % creates pseudoinverse of Es1, finds Phi
  zest=eig(TpT);     %take the eigenvalues of that matrix (This is eigenvalues of Phi) which are complex exponentials
  west=(log(zest(:))*j).'; %take natural log to find omega from exp(j*omega)
% if west is expected to be always real, uncomment next line
%   west=real(west);
  A=exp(j*i*west); % Mx1 matrix of complex exponentials comprising A
  Aest=A\y;        % find Aest = A\y
  
  erro=sum(abs(A*Aest-y).^2)/sum(abs(y).^2); %mean squared error
  
%   west = west*j; %not sure why this line is here



