function [avg_w, avg_error] = ten_fold_cross_validation(x, y, lambdas)
% 10-fold cross validation to find the best lambda
% For each lambda in lambdas array, perform 10 iterations of training with
% a data of 9:1 ratio (training:testing) then find the average error and
% weight array of those tens iterations of that lambda.

% calculate size of inputs
lambda_length = size(lambdas);
y_data_length = size(y);
x_data_size = size(x);

% initialize avg weight array and error array
avg_w = NaN(x_data_size(2)+1, lambda_length(2));
avg_error = NaN(2, lambda_length(2));

% loop through every lambda value
for i=1:lambda_length(2)
   
    % initialize temp weight array and error array for this lambda
    temp_w = NaN(x_data_size(2)+1, 10);
    temp_error = NaN(10, 2);
    
    % loop ten times to get averages
    for j=1:10
        
        % group x and y dataset into one
        data_set  = [x y];
        
        % randomly shuffle dataset
        random_data_set = data_set(randperm(size(data_set, 1)), :);
        
        % first 90% of randomly shuffled data is training
        training_data = random_data_set(1:(y_data_length(1)*0.9),:);
        
        % last 10% of randomly shuffled data is validation
        validation_data = [ones((y_data_length(1)*0.1), 1), ...
            random_data_set((y_data_length(1)*0.9)+1:end,:)];
        
        % linear regression learning to get weight vector from training
        % data
        temp_w(:,j) = linreg(training_data(:, 1:x_data_size(2)), ...
            training_data(:,x_data_size(2)+1), lambdas(i));
        
        % calculate in sample error using training data
        error_count = 0;
        for p=1:(y_data_length(1)*0.9)
            error_count = error_count + (sign(dot(temp_w(:,j), ...
                [1, training_data(p, 1:x_data_size(2))])) ...
                ~= training_data(p, x_data_size(2)+1));
        end
        temp_error(j, 1) = error_count/(y_data_length(1)*0.9);

        % calculate out of sample error using validation data
        error_count = 0;
        for p=1:(y_data_length(1)*0.1)
            error_count = error_count + (sign(dot(temp_w(:,j), ...
                validation_data(p, 1:x_data_size(2)+1))) ...
                ~= validation_data(p, x_data_size(2)+2));
        end
        temp_error(j, 2) = error_count/(y_data_length(1)*0.1);

    end
    
    % average out the ten iterations of each lambda, we should get seven
    % weight vectors and out of sample error estimates because there are
    % seven lambdas to test. More would occur if more lambdas are used.
    avg_w(:, i) = mean(temp_w, 2);
    avg_error(:, i) = (100.*mean(temp_error))'; 
end
end

