function [bestw, besterror] = pocket(x, y)
% pocket: trains pla and returns final weighted vector and iterations it took 
% INPUTS:
%   x: training data
%   y: label data
%   nu: learning rate
%**************************************************************************

% Initial weight vector
w = (-1+2*rand(3, 1));
bestw = w;

% Get length of training data
N = length(x);

% calculate initial w error
besterror = 0;
for i=1:N
    besterror = besterror + (sign(dot(bestw,x(i,:))) ~= y(i));
end
besterror = besterror/N;

% iterations for PLA
for k=1:50
    
    % one PLA iteration
    % get classifications based on hypothesis
    y_hyp = zeros(N,1);
    for i = 1:N
        y_hyp(i) = sign(dot(w,x(i,:)));
    end

    % compare classifications to training data
    comp_arr = (y_hyp == y);
    if y_hyp == y
        break;      % just in case training data can be fully separable
    end
   
    % update weight vector based on first misclassified data point
    for i=1:N
        if comp_arr(i)==0
            w = w + y(i)*x(i,:)';
            break;
        end
    end
    
    % calculate in sample error
    temperror = 0;
    for i=1:N
        temperror = temperror + (sign(dot(w,x(i,:))) ~= y(i));
    end
    temperror = temperror/N;
    
    % update bestw
    if temperror < besterror
        bestw = w;
        besterror = temperror;
    end
end