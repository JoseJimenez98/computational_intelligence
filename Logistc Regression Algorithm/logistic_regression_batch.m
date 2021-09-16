function [w, current_error] = logistic_regression_batch(x, y, nu)
% batch: trains logistic regression and returns final weighted vector
% INPUTS:
%   x: training data
%   y: label data
%   nu: learning rate
%**************************************************************************

% Initial weight vector
w = (-1+2*rand(3, 1));

% Get length of training data
N = length(x);

% calculate initial w error
past_error = 0;
for i=1:N
    past_error = past_error + (sign(dot(w,x(i,:))) ~= y(i));
end
past_error = past_error/N;

% initialize batch grad array
grad = NaN(N, 3);

% iterations for logistic regression
stop_flag = 0;
while ~(stop_flag)
    
    % calculate grad for each data point
    for i=1:N
        grad(i, :) = (y(i)*x(i,:))/(1+exp((y(i)*dot(w, x(i, :)))));
    end
    
    % calculate average direction vector
    v = (sum(grad)/N)';
    
    % update wieght vector
    w = w + (nu)*v;
    
    % update stopping criteria using error
    % calculate in sample error
    current_error = 0;
    for i=1:N
        current_error = current_error + (sign(dot(w,x(i,:))) ~= y(i));
    end
    current_error = current_error/N;
    
    % decide whether to stop
    if (abs(past_error-current_error) < 0.01)
        stop_flag = 1;
    else
        past_error = current_error;
    end    
end