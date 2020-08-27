%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% See the effect of only reading 2 sensors without combining
% Compared to single sensor reading
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc;more off;
%% ---------------------------------
% Trajectory
%-----------------------------------
dt = 0.01;
x = 0:dt:20*pi;
y = 10*sin(x).*exp(-x/12);

noise1 = normrnd(0, 1, size(x));
noise2 = normrnd(0, 1, size(x));

y1_noisy = y + noise1;
y2_noisy = y + noise2;

[~,len] = size(x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           SINGLE READING PART
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% initiate kalman filter matrices
%state vector initial guess
%      [position
x_hat = [5];
% state matrix
A = [1];
%input vector
B = 0;
u = 0;
%output scale
C =[1]; %we have access only to current position
% inital covatiance estimation
P = eye(size(A))*0.5;
%measurement covariance matrix
%IF R=0 THEN K=1; (ajust primarily with measurment update) 
%IF R=large THEN K=0; (ajust primarily with predicted state)
R=[10];
%process noise covariance matrix
%(keep covariance matrix P from going too small or going to 0)
%IF P=0 THEN measurment update is ignored
Q = eye(size(A))*0;

%% Iteration
x_avg = [];
x_hat_ar = [];
y_meas = [0];
for i = 1:len
    y_meas(1) = y1_noisy(i);
    
    [x_p, P_p] = predict(A,x_hat,B,u,P,Q);
    [x_hat, P] = update(x_p, P_p, C, y_meas, R);
    
    x_hat_ar = [x_hat_ar x_hat];
end
sceme1 = x_hat_ar;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           DOUBLE READING PART
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% initiate kalman filter matrices
%state vector initial guess
%       position 1
%       position 2
x_hat = [5;
         -5];
% state matrix
A = [1 0;
     0 1];
%input vector
B = 0;
u = 0;
%output scale
C =[1 0;
    0 1]; %we have access only to current position
% inital covatiance estimation
P = eye(size(A))*0.5;
%measurement covariance matrix
%IF R=0 THEN K=1; (ajust primarily with measurment update) 
%IF R=large THEN K=0; (ajust primarily with predicted state)
R=[10 0;
    0 10];
%process noise covariance matrix
%(keep covariance matrix P from going too small or going to 0)
%IF P=0 THEN measurment update is ignored
Q = eye(size(A))*0;


%% Iteration
x_avg = [];
x_hat_ar = [];
y_meas = [0;0];
for i = 1:len
    y_meas(1) = y1_noisy(i);
    y_meas(2) = y2_noisy(i);
            
    [x_p, P_p] = predict(A,x_hat,B,u,P,Q);
    [x_hat, P] = update(x_p, P_p, C, y_meas, R);
    
    x_hat_ar = [x_hat_ar x_hat];
end

sceme2 = x_hat_ar(1,:);

%% PLOT

figure(4);
plot(x, y,'y',x, sceme2, 'r',x, sceme1, 'g', 'LineWidth', 2, 'MarkerSize', 4), xlabel('t'), ylabel('f(t)'), axis equal, grid on, hold on;
title(['1 sensor vs 2 sensor (not combined)']);
legend({'ground truth', '2 sensor', '1 sensor'},'Location','northeast')
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