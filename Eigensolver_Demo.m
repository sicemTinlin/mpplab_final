%% create an n by n hermitian matrix with arbitrary complex values, then solve on CPU and GPU

clear all
close all

max_value = 5;
use_rand = 0;

min_n = 2;
max_n = 4000;
npoints = 40;

cpu_times = zeros(1, npoints);
gpu_double_times = zeros(1, npoints);
gpu_double_total_times = zeros(1,npoints);
gpu_single_times = zeros(1, npoints);
gpu_single_total_times = zeros(1,npoints);

cpu_error = zeros(1,npoints);
gpu_double_error = zeros(1,npoints);
gpu_single_error = zeros(1,npoints);

sizes = ceil(linspace(min_n,max_n,npoints));

for n = 1:length(sizes)
    %% get random hermitian matrix and solve for eig in 3 ways
    H = get_hermitian(sizes(n),max_value,use_rand);
    
    disp(n)
    
    %Solve for eigenvalues and eigenvectors on CPU and time:
    cpu_t = tic;
    [cpu_v,cpu_d] = eig(H);
    cpu_times(n) = toc(cpu_t);
    
    % allocate H on GPU (leave as double)
    gpu_double_total_t = tic; %use if including memory overhead
    H_d_double = gpuArray(H);
    %    gpu_double_t = tic; %use if ignoring memory overhead
    [gpu_v_double, gpu_d_double] = eig(complex(H_d_double));
    %    gpu_double_times(n) = toc(gpu_double_t);
    [gpu_v_double_h, gpu_d_double_h] = gather(gpu_v_double,gpu_d_double);
    gpu_double_total_times(n) = toc(gpu_double_total_t);
    
    % allocate H on GPU (specify as single)
    gpu_single_total_t =tic; %use if including memory overhead
    H_d_single = gpuArray(single(H));
    %     gpu_single_t = tic; %use if ignoring memory overhead
    [gpu_v_single, gpu_d_single] = eig((H_d_single));
    %     gpu_single_times(n) = toc(gpu_single_t);
    [gpu_v_single_h, gpu_d_single_h] = gather(gpu_v_single,gpu_d_single);
    gpu_single_total_times(n) = toc(gpu_single_total_t);
    
    %% check that eigenvalues calculated are valid (H*V) = (V*D) within tolerance
    
    zero_mat = zeros(sizes(n),'double');
    zero_mat = complex(zero_mat,0);
    
    %calculate CPU error
    t1 = H*cpu_v;
    t2 = cpu_v * cpu_d;
    r = t1-t2;
    cpu_error(n) = immse(zero_mat,r);
    
    % calculate GPU double precision error
    t1 = H*gpu_v_double_h;
    t2 = gpu_v_double_h*gpu_d_double_h;
    r = t1-t2;
    %find mse
    gpu_double_error(n) = immse(zero_mat,r);
    
    zero_mat = zeros(sizes(n),'single');
    zero_mat = complex(zero_mat,0);
    
    %calculate GPU single precision error
    t1 = H*gpu_v_single_h;
    t2 = gpu_v_single_h*gpu_d_single_h;
    r = t1-t2;
    % find mse
    gpu_single_error(n) = immse(zero_mat,r);
    
end



%plot results

%for plotting if ignoring memory overhead
% plot(sizes,cpu_times,'-r',sizes,gpu_double_times,'-b',sizes,gpu_single_times,'-g')
% title('eig() performance on CPU vs. GPU double and single precision')
% ylabel('time(s)')
% xlabel('length of Hermitian (square) matrix')
% legend('CPU', 'GPU Double Precision', 'GPU Single Precision')

%for plotting if including memory overhead
subplot(2,1,1)
plot(sizes,cpu_times,'-r',sizes,gpu_double_total_times,'-b',sizes,gpu_single_total_times,'-g')
title('eig() performance on CPU vs. GPU double and single precision')
ylabel('execution time (s)')
xlabel('length of Hermitian (square) matrix')
legend('CPU', 'GPU Double Precision', 'GPU Single Precision')
legend('Location','northeastoutside')
subplot(2,1,2)
semilogy(sizes,cpu_error,'-r',sizes,gpu_double_error,'-b',sizes,gpu_single_error,'-g')
title('MSE on CPU vs. GPU double and single precision')
xlabel('length of Hermitian (square) matrix')
ylabel('mean squared error')
legend('CPU', 'GPU Double Precision', 'GPU Single Precision')
legend('Location','northeastoutside')