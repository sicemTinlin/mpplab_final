function [test_array,t] = esprit_data_gen(n_samples, n_tests, n_signals, use_rand)

if use_rand == 0
    sd = 6158;
    rand('seed',sd);
end

f_max = 1e6;
A_max = 10;

test_array = zeros(n_samples,n_tests);

for test_index = 1:n_tests
    f = sort(rand(1,n_signals)*f_max);
    w = 2*pi*f;
    A = rand(1,n_signals)*A_max;

    T_min = 1/f_max;
    samples_per_period = 100;
    T_sample = T_min/samples_per_period;

    t = T_sample*(0:1:n_samples-1);

    % let our signal x(t) = A*exp(-j*w*t) = A*(cos(w*t)-j*sin(w*t))

    x = zeros(n_samples,n_signals);

    for j = 1:n_signals
        for i = 1:n_samples
            x(i,j) = A(j)*exp(1i*w(j)*t(i));
        end
    end
    
    y = zeros(n_samples,1);

    for i = 1:n_signals
    y = y(:) + x(:,i);
    end
    
    test_array(:,test_index) = y(:,1);
    
end
    