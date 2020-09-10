clear;clc;more off;
%% ---------------------------------
% Trajectory
%-----------------------------------
dt = 0.01;
x = 0:dt:20*pi;
y = 10*sin(x).*exp(-x/12);

noise1 = normrnd(0, 2, size(x));
noise2 = normrnd(0, 2, size(x));
noise3 = normrnd(0, 2, size(x));
noise4 = normrnd(0, 2, size(x));
noise5 = normrnd(0, 2, size(x));

y1_noisy = y + noise1;
y2_noisy = y + noise2;
y3_noisy = y + noise3;
y4_noisy = y + noise4;
y5_noisy = y + noise5;

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
         0.1];
% state matrix
A = [1  dt;
     0  1];
%input vector
B = 0;
u = 0;
%output scale
C =[1 dt]; %we have access only to current position
% inital covatiance estimation
P = eye(size(A))*0.5;
%measurement covariance matrix
%IF R=0 THEN K=1; (ajust primarily with measurment update) 
%IF R=large THEN K=0; (ajust primarily with predicted state)
R=[1];
%process noise covariance matrix
%(keep covariance matrix P from going too small or going to 0)
%IF P=0 THEN measurment update is ignored
Q = eye(size(A))*0;


%% Iteration
x_hat_ar = [];
y_meas = [0;0;0;0;0];
for i = 1:len
    ratio = 0.5;
    %cobination part
    %the combined reding is the regarded as measured part
    %so 2 measured values are the same
    y_meas(1) = y1_noisy(i);
    y_meas(2) = y2_noisy(i);
    y_meas(3) = y3_noisy(i);
    y_meas(4) = y4_noisy(i);
    y_meas(5) = y5_noisy(i);

    %state priori prediction
    x_p = A*x_hat + B*u; %state predicted
    %Pior estimation of estimation covariance
    P_p = A*P*A' + Q; %estimation covariance
    P_p = diag(diag(P_p)); %take only diagonal part of P
    
    for i=1:length(y_meas)
        % Measurement error
        v = y_meas(i) - C*x_p;
        % S the measurement prediction covariance on the time step k
        S = C*(P_p)*C' + R;
        % ----------------------------
        %calculate Filter gain
        K = P_p*C'/S;
    end
    % state posterior prediction
    x_hat = x_p + K*(v);
    % Posterior estimation of estimation covariance
    P = (eye(size(P_p)) - K*C)*P_p;
    
    x_hat_ar = [x_hat_ar x_hat];
end
x_hat_ar_2sen = x_hat_ar(1,:);

%% ----------Plot---------------
figure(5);
plot(x, y, 'y', x, x_hat_ar_2sen, 'r', 'LineWidth', 1, 'MarkerSize', 4), xlabel('t'), ylabel('f(t)'), axis equal, grid on, hold on;
title(['5 sensor combined']);
legend({'ground truth', '5 sensors'},'Location','northeast')
set(gca,'FontSize',12);
set(gca,'FontName','serif');
set(gca,'FontWeight','bold');
set(gca,'LineWidth',2);