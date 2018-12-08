
function [west_h,Aest_h,erro_h]= gpu_esprit_v3(y_h,M_h,Ma_h)
  M_d = gpuArray(single(M_h));
  Ma_d = gpuArray(single(Ma_h));
  j_h = 1i;
  j_d = gpuArray(single(j_h));
  y_h=y_h(:);          %make sure y is a row vector
  y_d = gpuArray(single(complex(y_h)));
  N_d = length(y_d);     %get the number of elements             
  i_d = gpuArray(single((1:N_d)'));        %get a column vector of equal length to i
  x_d = hankel(complex(y_d));     %make a hankel matrix out of the data
  sub_x_d = x_d(1:M_d,1:N_d-M_d+1);
  x_d = flipud(complex(sub_x_d)); %Take the hankel matrix of the data out to row M and column M+1 and discard the rest
  
  Rx_d = complex(complex(x_d)*complex(x_d'))/ complex(N_d - M_d+1); %find the sample covariance matrix of the new data

  D_d = eig(complex(Rx_d));
  D_d = real(D_d);  
  [~,ind_d] = min(D_d); %find the minimum real eigenvalue of Rx
  thr_d = D_d(ind_d);      % store that minimum eigenvalue of Rx
  
  diag_h = eye(M_h); %make identity matrix on host
  diag_d = gpuArray(single(diag_h)); %transfer to device
  
  Rx_d = complex(Rx_d) - complex(thr_d)* complex(diag_d); % find ASA' (see complex cisoids in noise paper eq.14)
  
  [V_d,D_d] = eig(complex(Rx_d));    % find the generalized eigenvalues and eigenvectors of ASA' (also called Cxx in cisoids and noise paper)
  
  diag_D_d = diag(complex(D_d));
  
  D_d = real(diag_D_d);  % takes the eigenvalues from diagonal matrix D
  
  Ds_d = sort(complex(D_d));       % sort eigenvalues of Cxx
  
  subcalc1 = complex(Ds_d(M_d-Ma_d));
  thr_d = abs(complex(subcalc1)); 
  
  ind_d = find(complex(D_d) > complex(thr_d));    %get all eigenvalues greater than that eigenvalue chosen above
  d_d = gpuArray(single(length(ind_d)));    % get how many eigenvalues there are that are greater than the eigenvalue chosen above
  Es_d = complex(V_d(:,ind_d));      % get the first d column vectors of V (which are the first d eigenvectors of Cxx)
  
  if (d_d==M_d)    % d must not equal M
    disp('error: d_d=M_d!');
  end
  
  Es1_d = complex(Es_d(1:M_d-1,:));  % submatrix of eigenvector matrix
  Es2_d = complex(Es_d(2:M_d,:));    % submatrix of eigenvector matrix
  TpT_d = complex(pinv(complex(Es1_d)))*complex(Es2_d); % creates pseudoinverse of Es1, finds Phi
  zest_d = eig(complex(TpT_d));     %take the eigenvalues of that matrix (This is eigenvalues of Phi) which are complex exponentials
  west_d=complex((complex(log(complex(zest_d(:))))*complex(j_d))).'; %take natural log to find omega from exp(j*omega)
% if west is expected to be always real, uncomment next line
%   west=real(west);
  A_d = exp(complex(j_d)*complex(i_d)*complex(west_d)); % Mx1 matrix of complex exponentials comprising A
  Aest_d = complex(A_d)\complex(y_d);        % find Aest = A\y
  
  erro_d = sum(abs(complex(A_d)*complex(Aest_d)-complex(y_d)).^2)/sum(abs(complex(y_d)).^2); %mean squared error
  
  
  erro_h = gather(erro_d);
  Aest_h = gather(complex(Aest_d));
  west_h = gather(complex(west_d));




