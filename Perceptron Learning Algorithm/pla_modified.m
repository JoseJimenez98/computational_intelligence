function [w, t] = pla_modified(x, y, nu)
% pla: trains pla and returns final weighted vector and iterations it took 
% INPUTS:
%   x: training data
%   y: label data
%**************************************************************************

% Initial weight vector
w = (-1+2*rand(3, 1));

% Initialize iterations
t = 1;

% Get length of training data
N = length(x);

for k=1:1000
    % get classifications based on hypothesis
    y_hyp = zeros(N,1);
    for i = 1:N
        y_hyp(i) = sign(dot(w,x(i,:)));
    end

    % compare classifications to training data
    comp_arr = (y_hyp == y);
    if y_hyp == y
        break;
    end
    
    % update weight vector based on first misclassified data point
    for i=1:N
        if comp_arr(i)==0
            
            % calculate signal s(t)
            s = dot(w, x(i,:));
            
            % update weight vector
            w = w + (nu*(y(i)-s)*x(i,:)');
            t = t + 1;
            break;
        end
    end
end