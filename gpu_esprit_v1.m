
function [west,Aest,erro]=gpu_esprit_v1(y,M,Ma)
% input: vector y (complex in general)
%        M : order (max. number of complex exponentials expected)
%            M<N/2!
% y is supposed to be composed of terms given by {Aest*exp(j*west*n)}
%                                                 n=1:N
% output: west: complex in general 
%         Aest: complex in general
%         erro: is the mean squared error in energy obtained
% NOTE: west*n = w*d so w = west*(n/d)
  M = gpuArray(M);
  Ma = gpuArray(Ma);
  j = gpuArray(1i);
  
  y= gpuArray(complex(y(:)));          %make sure y is a row vector
  N = gpuArray(length(y));     %get the number of elements             
  i=(1:N)'; %get a column vector of equal length to i
  i = gpuArray(i);
  x=hankel(complex(y));     %make a hankel matrix out of the data
  x = flipud(complex(x(1:M,1:(N-M+1)))); %Take the hankel matrix of the data out to row M and column M+1 and discard the rest
  Rx = complex(x)*complex(x)'/(N-M+1); %find the sample covariance matrix of the new data

  D = real(eig( complex(Rx))); %find the real eigenvalues of Rx
  [~,ind]=min(D); %find the minimum real eigenvalue of Rx
  thr = D(ind);      % store that minimum eigenvalue of Rx
 
  Rx = complex(Rx) - thr*eye(M); % find ASA' (see complex cisoids in noise paper eq.14)
  [V,D]=eig(complex(Rx));    % find the generalized eigenvalues and eigenvectors of ASA' (also called Cxx in cisoids and noise paper)
  D = real(diag(complex(D)));  % takes the eigenvalues from diagonal matrix D
  Ds = sort(D);       % sort eigenvalues of Cxx
  thr = abs(Ds(M-Ma)); 
  ind = find(D>thr);    %get all eigenvalues greater than that eigenvalue chosen above
  d = length(ind);    % get how many eigenvalues there are that are greater than the eigenvalue chosen above
  Es = complex(V(:,ind));      % get the first d column vectors of V (which are the first d eigenvectors of Cxx)
  if (d==M),        % d must not equal M
    disp('error: d=M!');
  end;
  Es1 = complex(Es(1:M-1,:));  % submatrix of eigenvector matrix
  Es2 = complex(Es(2:M,:));    % submatrix of eigenvector matrix
  TpT = pinv(complex(Es1))*complex(Es2); % creates pseudoinverse of Es1, finds Phi
  zest = eig(complex(TpT));     %take the eigenvalues of that matrix (This is eigenvalues of Phi) which are complex exponentials
  west = (log(complex(zest(:)))*j).'; %take natural log to find omega from exp(j*omega)
% if west is expected to be always real, uncomment next line
%   west=real(west);
  A = exp(j*i*complex(west)); % Mx1 matrix of complex exponentials comprising A
  Aest = complex(A)\complex(y);        % find Aest = A\y
  
  erro = sum(abs(A*complex(Aest)-complex(y)).^2)/sum(abs(complex(y)).^2); %mean squared error
  
  west = gather(complex(west));
  Aest = gather(complex(Aest));
  erro = gather(erro);
  
%   west = west*j; %not sure why this line is here



