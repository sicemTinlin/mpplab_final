%% generate sinusoids of random frequency and amplitude
clear all
close all

% sd = 6158;
% rand('seed',sd);

f_max = 1e6;
A_max = 10;
n_signals = 3;

f = sort(rand(1,n_signals)*f_max);
w = 2*pi*f;
A = rand(1,n_signals)*A_max;

T_max = 1/f(1);
T_min = 1/f(end);
min_samples_per_period = 100;
T_sample = T_min/min_samples_per_period;

n_periods_min = 1; %for lowest freq
T_tot = n_periods_min*T_max;

n_samples = ceil(T_tot/T_sample);

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

%plot component signals
subplot(3,2,1)
    plot(t,real(x))
    title('real part of x');
    xlabel('time(s)')
    ylabel('Re(x)')
subplot(3,2,3)
    plot(t,abs(x))
    title('magnitude of x');
    xlabel('time(s)')
    ylabel('Abs(x)')
subplot(3,2,5)
    plot(t,angle(x)*(180/pi))
    title('phase of x')
    xlabel('time(s)')
    ylabel('Arg(x)')
%plot total signal
subplot(3,2,2)
    plot(t,real(y))
    title('real part of y')
    xlabel('time(s)')
    ylabel('Re(y)')
subplot(3,2,4)
    plot(t,abs(y))
    title('magnitude of y')
    xlabel('time(s)')
    ylabel('Abs(y)')
subplot(3,2,6)
    plot(t,angle(y)*(180/pi))
    title('phase of y')
    xlabel('time(s)')
    ylabel('Arg(y)')
    
%apply esprit algorithm to try and extract frequencies
model_order = 50; %must be less than length(y)/2
expected_sinusoids =3; %expected number of sinusoids
[west,Aest,erro] = esprit(y,model_order,expected_sinusoids);
 
% our estimated signal is Aest*exp(j*west*n)
% our real signal is A*exp(j*w*t)
% so west*n = w*t
% so w = west*(n/t)

west = west*(length(y)/T_tot);
fest = west/(2*pi)
Aest
erro