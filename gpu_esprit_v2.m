
function [west,Aest,erro]=gpu_esprit_v2(y,M,Ma)

  M = gpuArray(single(M)); %should not be complex
  Ma = gpuArray(single(Ma)); %should not be complex
  j = gpuArray(single(1i)); %complex implicitly
  y= gpuArray(complex(single(y(:))));         
  N = gpuArray(single(length(y))); %should not be complex
  i=complex(single((1:N)))'; 
  
  x=hankel(y);   
  x = flipud(x(1:M,1:(N-M+1)));
  Rx = x*x'/(N-M+1); 

  D = real(eig(Rx)); 
  [~,ind]=min(D);
  thr = D(ind);   
 
  Rx = complex(Rx) - thr*eye(M);
  [V,D]=eig(complex(Rx));   %problem spot 1
  D = real(diag(complex(D))); 
  Ds = sort(D);  
  thr = abs(Ds(M-Ma)); 
  ind = find(D>thr);  
  d = length(ind);    
  Es = complex(V(:,ind)); 
  if (d==M)      
    disp('error: d=M!');
  end
  Es1 = complex(Es(1:M-1,:));  
  Es2 = complex(Es(2:M,:));   
  TpT = pinv(complex(Es1))*complex(Es2); 
  zest = eig(complex(TpT));     %problem spot 2
  west = (log(complex(zest(:)))*j).'; 
  
% if west is expected to be always real, uncomment next line
%   west=real(west);
  A = exp(j*i*complex(west)); 
  Aest = complex(A)\complex(y);  
  erro = sum(abs(A*complex(Aest)-complex(y)).^2)/sum(abs(complex(y)).^2); %mean squared error
  
  west = gather(complex(west));
  Aest = gather(complex(Aest));
  erro = gather(erro);
end
  
