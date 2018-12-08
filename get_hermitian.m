function [H] = get_hermitian(n,max_val,use_rand)

if use_rand ~= 0
    sd = 6158;
    rng(sd);
else
    rng('shuffle');
end

%allocate matrix
H = zeros(n);

for i = 1:n
    for j = 1:n
        if i == j
            %allocate real diagonal element
            H(i,j) = rand * max_val;
        else
            %allocate H(i,j) and H(j,i)
            H(i,j) = (rand * max_val)+ 1i*(rand*max_val);
            H(j,i) = conj(H(i,j));
        end
    end
end

if ishermitian(H) ~= 1
  	error('Unable to create Hermitian matrix')
end

end