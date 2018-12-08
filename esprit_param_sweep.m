%% sweep esprit input values to generate test data

close all
clear all

n_tests = 10; % no need to change this one

% n_samples = ceil(linspace(1000,10000,10)); 
n_samples = 5000;

n_signals = 3; %default
%n_signals = [1 2 3 4 5]; %sweep setup 

use_rand = 0; %don't change while writing report for consistency

%use_GPU = 3; %0 for cpu, 1,2,... for version 1,2,... of GPU code
use_GPU = [3];
 
use_parPool = 0;

if use_parPool ~= 0 % 1 is smallest cluster, 6 is largest
    profiles = flip({'4core2thread','4core1thread','3core2thread','3core1thread','2core2thread','2core1thread'});
    if use_parPool > length(profiles)
        error('invalid profile selection')
    end
    cluster_profile = char(profiles(use_parPool));
    mycluster = parcluster(cluster_profile);
    if isempty(gcp('nocreate')) 
        mypool = parpool(cluster_profile);
    else
        delete(gcp('nocreate')) ; %admittedly inefficient, but effective
        mypool = parpool(cluster_profile);
    end
else
    delete(gcp('nocreate')); %kill any parallel pool if present
    cluster_profile = 'sequential';
end
    
%% use this if only running one case
% esprit_harness(n_tests,n_samples,n_signals,use_rand,use_GPU,use_parPool);

%% set this loop up to sweep various parameters

for m = 1:length(use_GPU)
    for n = 1:length(n_samples)
        esprit_harness(n_tests,n_samples(n),n_signals,use_rand,use_GPU(m),use_parPool,cluster_profile);
    end
end

%alarm for program end
load handel;
sound(y,Fs);