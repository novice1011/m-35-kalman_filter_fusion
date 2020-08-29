%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comparison between reading 2 states (combined) vs 1 state
% 2 states using complementary concept
% Case 1 state : 1. position sensor 1; 2. velocity prediction sensor 1; 3.
% position from sensor 2; 4. velocity prediction sensor 2
% Case 2 state : 1. position from sensor 1; 2. velocity prediction from
% sensor 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc;more off;
%% ---------------------------------
% Trajectory
%-----------------------------------
dt = 0.01;
x = 0:dt:20*pi;
y = 10*sin(x).*exp(-x/12);

noise1 = normrnd(0, 1, size(x));
noise2 = normrnd(0, 0.1, size(x));

y1_noisy = y + noise1;
y2_noisy = y + noise2;

[~,len] = size(x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           DOUBLE SENSOR PART
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% initiate kalman filter matrices
%state vector initial guess
%       position 1
%       position 2
%       velocity 1
%       velocity 2
x_hat = [5;
         -5;
         0.1;
         0.1];
% state matrix
A = [1  0   dt  0;
     0  1  0   dt;
     0  0   1   0;
     0  0   0   1];
%input vector
B = 0;
u = 0;
%output scale
C =[1 0 0 0;
    0 1 0 0;
    0 0 0 0;
    0 0 0 0]; %we have access only to current position
% inital covatiance estimation
P = eye(size(A))*0.5;
%measurement covariance matrix
%IF R=0 THEN K=1; (ajust primarily with measurment update) 
%IF R=large THEN K=0; (ajust primarily with predicted state)
R=[0.1 0 0 0;
   0 0.1 0 0;
   0 0  100 0;
   0 0  0 100];
%process noise covariance matrix
%(keep covariance matrix P from going too small or going to 0)
%IF P=0 THEN measurment update is ignored
Q = eye(size(A))*0;


%% Iteration
x_hat_ar = [];
y_meas = [0;0;0;0];
for i = 1:len
    ratio = 0.5;
    %cobination part
    %the combined reding is the regarded as measured part
    %so 2 measured values are the same
    y_meas(1) = (1-ratio)*y1_noisy(i) + (ratio)*y2_noisy(i);
    y_meas(2) = (1-ratio)*y1_noisy(i) + (ratio)*y2_noisy(i);
            
    [x_p, P_p] = predict(A,x_hat,B,u,P,Q);
    [x_hat, P] = update(x_p, P_p, C, y_meas, R);
    
    x_hat_ar = [x_hat_ar x_hat];
end
x_hat_ar_2sen = x_hat_ar(1,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           SINGLE SENSOR PART
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% initiate kalman filter matrices
%state vector initial guess
%       position 1
%       velocity 1
x_hat = [5;
         0.1];
% state matrix
A = [1  dt;
     0  1;];
%input vector
B = 0;
u = 0;
%output scale
C =[1 0;
    0 0]; %we have access only to current position
% inital covatiance estimation
P = eye(size(A))*0.5;
%measurement covariance matrix
%IF R=0 THEN K=1; (ajust primarily with measurment update) 
%IF R=large THEN K=0; (ajust primarily with predicted state)
R=[0.1 0;
   0 100];
%process noise covariance matrix
%(keep covariance matrix P from going too small or going to 0)
%IF P=0 THEN measurment update is ignored
Q = eye(size(A))*0;


%% Iteration
x_hat_ar = [];
y_meas = [0;0];
for i = 1:len
    y_meas(1) = y1_noisy(i);
            
    [x_p, P_p] = predict(A,x_hat,B,u,P,Q);
    [x_hat, P] = update(x_p, P_p, C, y_meas, R);
    
    x_hat_ar = [x_hat_ar x_hat];
end
x_hat_ar_1sen = x_hat_ar(1,:);

%% ----------Plot---------------
figure(4);
plot(x, y, 'y', x, x_hat_ar_2sen, 'r', x, x_hat_ar_1sen, 'g', 'LineWidth', 1, 'MarkerSize', 4), xlabel('t'), ylabel('f(t)'), axis equal, grid on, hold on;
title(['1 sensor vs 2 sensor (combined complementary inspired)']);
legend({'ground truth', '2 sensors', '1 sensor'},'Location','northeast')
set(gca,'FontSize',12);
set(gca,'FontName','serif');
set(gca,'FontWeight','bold');
set(gca,'LineWidth',2);

%% prediction function
function [x_p, P_p] = predict(A,x_hat,B,u,P,Q)
    %state priori prediction
    x_p = A*x_hat + B*u; %state predicted
    %Pior estimation of estimation covariance
    P_p = A*P*A' + Q; %estimation covariance
    P_p = diag(diag(P_p)); %take only diagonal part of P
end

%% update function
function [x_hat, P] = update(x_p, P_p, C, y, R)
    % is the innovation or the measurement residual on time step k.
    % Measurement error
    v = y - C*x_p;
    % S the measurement prediction covariance on the time step k
    S = C*(P_p)*C' + R;
    % ----------------------------
    %calculate Filter gain
    K = P_p*C'/S;
    % state posterior prediction
    x_hat = x_p + K*(v);
    % Posterior estimation of estimation covariance
    P = (eye(size(P_p)) - K*C)*P_p;
end

%https://arxiv.org/pdf/1204.0375.pdf
%https://www.youtube.com/watch?v=jn8vQSEGmuM
%http://ros-developer.com/2019/04/10/kalman-filter-explained-with-python-code-from-scratch/