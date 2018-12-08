function [] = esprit_harness(n_tests, n_samples, n_signals, use_rand, use_GPU, use_parPool,cluster_profile)

%% generate test data for esprit program

t_start = tic;
[data_array, t] = esprit_data_gen(n_samples,n_tests,n_signals,use_rand);
setup_time = toc(t_start);


%% run test data through esprit program

model_order = ceil(0.4*n_samples);
expected_sinusoids = n_signals;

Aest = zeros(n_signals,n_tests);
w_est = zeros(n_signals,n_tests);
erro = zeros(1,n_tests);

run_times = zeros(n_tests,1);

%suppress deficient matrix warning (these will be caught by the error check)
warn_id = 'MATLAB:rankDeficientMatrix';
warning('off',warn_id);

if use_parPool == 0
    for n = 1:n_tests
        
        if use_GPU == 0
            t_start_sub = tic;
            [w_est(:,n),Aest(:,n),erro(1,n)] = esprit(data_array(:,n),model_order,expected_sinusoids);
            run_times(n) = toc(t_start_sub);
            %uncomment to find the true frequency estimation
            %f_est(:,n) = real(w_est(:,n)*(length(data_array(:,n))/t(end))/(2*pi));
        elseif use_GPU == 1
            t_start_sub = tic;
            [w_est(:,n),Aest(:,n),erro(1,n)] = gpu_esprit_v1(data_array(:,n),model_order,expected_sinusoids);
            run_times(n) = toc(t_start_sub);
        elseif use_GPU == 2
            t_start_sub = tic;
            [w_est(:,n),Aest(:,n),erro(1,n)] = gpu_esprit_v2(data_array(:,n),model_order,expected_sinusoids);
            run_times(n) = toc(t_start_sub);
        elseif use_GPU == 3
            t_start_sub = tic;
            [w_est(:,n),Aest(:,n),erro(1,n)] = gpu_esprit_v3(data_array(:,n),model_order,expected_sinusoids);
            run_times(n) = toc(t_start_sub);
        elseif use_GPU == 4
            t_start_sub = tic;
            [w_est(:,n),Aest(:,n),erro(1,n)] = gpu_esprit_v4(data_array(:,n),model_order,expected_sinusoids);
            run_times(n) = toc(t_start_sub);
        elseif use_GPU == 5
            t_start_sub = tic;
            [w_est(:,n),Aest(:,n),erro(1,n)] = gpu_esprit_v5(data_array(:,n),model_order,expected_sinusoids);
            run_times(n) = toc(t_start_sub);
        elseif use_GPU == 6
            t_start_sub = tic;
            myGPU = parallel.gpu.GPUDevice.current();
            [w_est(:,n),Aest(:,n),erro(1,n)] = gpu_esprit_v6(data_array(:,n),model_order,expected_sinusoids,myGPU);
            run_times(n) = toc(t_start_sub);
        end
    end
    processing_time = sum(run_times);
else
    true_total = tic;
    parfor n = 1:n_tests
        if use_GPU == 0
            t_start_sub = tic;
            [w_est(:,n),Aest(:,n),erro(1,n)] = esprit(data_array(:,n),model_order,expected_sinusoids);
            run_times(n) = toc(t_start_sub);
            %uncomment to find the true frequency estimation
            %f_est(:,n) = real(w_est(:,n)*(length(data_array(:,n))/t(end))/(2*pi));
        elseif use_GPU == 6
            t_start_sub = tic
            myGPU = parallel.gpu.GPUDevice.current();
            [w_est(:,n),Aest(:,n),erro(1,n)] = gpu_esprit_v6(data_array(:,n),model_order,expected_sinusoids,myGPU);
            run_times(n) = toc(t_start_sub);
        else
            error('only use_GPU 0 and 6 are supported for parfor')
        end
    end
    sum_runtimes = sum(run_times);
    true_total = toc(true_total);
end


%% print results

% remove failed runs so that error calculations are not distorted
%set max allowable error
max_erro = 0.001;
bad_runs = find(isnan(erro) | erro > max_erro);
bad_runs_str = sprintf('%d ', bad_runs);
erro(bad_runs) = [];

if use_rand == 0
    rand_str = 'rand_off';
else
    rand_str = 'rand_on';
end

myGPU = gpuDevice;

if use_GPU == 0
    accel_str = 'no_GPU';
else
    accel_str = strcat(myGPU.Name, ' V' , num2str(use_GPU));
end

if use_parPool == 0
    par_str = cluster_profile;
else
    par_str = strcat('ParPool_', cluster_profile);
end

fileName = strcat(par_str,{' '},accel_str,{' '}, num2str(n_tests), '_tests', {' '}, ...
    num2str(n_samples), '_samples', {' '}, num2str(n_signals),...
    '_signals', {' '} , rand_str, '.txt');
outputFile = fileName{1};
fileId = fopen(outputFile, 'w');

fprintf(fileId, 'Setup Info:\r');
fprintf(fileId, 'n_tests = %i\r',n_tests);
fprintf(fileId, 'n_samples = %i\r', n_samples);
fprintf(fileId, 'n_signals = %i\r', n_signals);
fprintf(fileId, 'model_order = %i\r', model_order);
fprintf(fileId, 'expected_sinusoids = %i\r', expected_sinusoids);
fprintf(fileId, 'use_rand = %i\r', use_rand);
fprintf(fileId, 'accelerate_flag = %i\r', use_GPU);

fprintf(fileId,'\r\nError info (failed runs excluded): \r');
fprintf(fileId,'max error = %.1e\r', max(erro));
fprintf(fileId,'min error = %.1e\r', min(erro));
fprintf(fileId,'mean error = %.1e\r',mean(erro));
fprintf(fileId,'median error = %.1e\r', median(erro));
fprintf(fileId,'mode error = %.1e\r', mode(erro));
fprintf(fileId,'failed runs (error == NaN or > %.1e) = %i\r', max_erro, length(bad_runs));

if length(bad_runs) >= 1
    fprintf(fileId,'failed run indices : %s\r',bad_runs_str);
end

fprintf(fileId,'\r\nTiming info (failed runs included): \r');
fprintf(fileId,'max run time = %.1e\r', max(run_times));
fprintf(fileId,'min run time = %.1e\r', min(run_times));
fprintf(fileId,'mean run time = %.1e\r',mean(run_times));
fprintf(fileId,'median run time = %.1e\r', median(run_times));
fprintf(fileId,'mode run time = %.1e\r', mode(run_times));
if use_parPool == 0
    fprintf(fileId,'total run time (all cycles) = %.1e\r', processing_time);
else
    fprintf(fileId,'total run time (parallel execution) = %.1e\r', true_total);
    fprintf(fileId,'sum of each string runtime = %.1e\r', sum_runtimes);
end

if length(bad_runs) >= 1
    plot(t,real(data_array(:,bad_runs)));
    title('Failed Data Sets')
end

% based on experience, bad runs do not appear to take longer than
% successful runs. Uncomment if verfication is needed
% run_times(bad_runs) = [];
% fprintf('\nTiming info (bad runs excluded): \r')
% fprintf('max run time = %.1e\r', max(run_times))
% fprintf('min run time = %.1e\r', min(run_times))
% fprintf('mean run time = %.1e\r',mean(run_times))
% fprintf('median run time = %.1e\r', median(run_times))
% fprintf('mode run time = %.1e\r', mode(run_times))
% fprintf('total run time (all cycles) = %.1e\r', processing_time)

fclose(fileId);