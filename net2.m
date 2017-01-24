function [weights_cell,bias_cell,train_results,test_results,validate_results] = net2(inputs,targets,nodelayers, numEpochs, batchSize, eta, lambda, momentum, transfer, cost, split, weights, bias)

% establish empty matricies that will be used to return the accuracy and
% cost on each partition
train_results = [];
test_results = [];
validate_results = [];

% randomly shuffle the initial inputs and training so that our test and
% validation sets will be representative
inputs_shuffle = randperm(size(inputs,2));

inputs = inputs(:,inputs_shuffle);
targets = targets(:,inputs_shuffle);

% assign the training, test, and validation data
    % split the training data
train_input = inputs(:,1:size(inputs,2) * split(1) / 100);
test_input = inputs(:,(size(inputs,2) * split(1) / 100) + 1:size(inputs,2) * (split(1) + split(2))/100);
validate_input = inputs(:,(size(inputs,2) * (split(1) + split(2))/100) + 1:size(inputs,2));

    % split the targets
train_target = targets(:,1:size(targets,2) * split(1) / 100);
test_target = targets(:,(size(targets,2) * split(1) / 100) + 1:size(targets,2) * (split(1) + split(2))/100);
validate_target = targets(:,(size(targets,2) * (split(1) + split(2))/100) + 1:size(targets,2));
 

% initialize the weights (if not provided in function call)
if exist('weights') == 0
    weights_cell = cell(1,size(nodelayers,2)-1);
    for l = 1:size(nodelayers,2)-1
        % mean of 0 and a s.d. of one divided by the square root of the
        % number of input weights (aka the # of nodes in the previous
        % layer)
        weights_cell{l} = (1/sqrt(nodelayers(l))) .* randn(nodelayers(l+1),nodelayers(l));
    end
else 
    weights_cell = weights
end

% randomly assign the biases (if not provided in function call)
if exist('bias') == 0
    bias_cell = cell(1,size(nodelayers,2)-1);
    for l = 1:size(nodelayers,2)-1
        bias_cell{l} = randn(nodelayers(l+1),1);
    end
else 
    bias_cell = bias
end

% write the column headers for our output
fprintf('   |         TRAIN                    ||         TEST               ||     VALIDATION\n')
fprintf('--------------------------------------------------------------------------\n')
fprintf('Ep | Cost     | Corr   | Acc          || Cost    | Corr | Acc       || Cost     | Corr | Acc\n')

% BEGIN BACKPROPOGATION ALGORITHM

% initiate outer loop for the total number of epochs
epoch = 1;
COST_validate_increase_counter = 0; % used to keep track of the cost on the validation data for early stopping
while epoch <= numEpochs

    % shuffle the training set so that the minibatches will be shuffled for
    % each epoch
    inputs_shuffle = randperm(size(train_input,2));

    train_input = train_input(:,inputs_shuffle);
    train_target = train_target(:,inputs_shuffle);
    
    % initialize the correct guesses and COST for the epopch to zero (used for
    % calculating accuracy)
    correct_guess_train = 0;
    
    COST_train = 0;
    
    % create an empty cell to hold the activations
    activation_cell = cell(1,size(nodelayers,2));
    
    % create an empty cell to hold the velocities for each weight - used
    % for momemntum calculation
    velocity_cell = cell(1,size(nodelayers,2)-1);
    % initialize all velocities to zero
    for l = 1:size(nodelayers,2)-1
        velocity_cell{l} = zeros(nodelayers(l+1),nodelayers(l));
    end

    % create an empty cell to hold the errors
    errors_cell = cell(1,size(nodelayers,2)-1);

    % create an empty cell to hold the weighted inputs
    z_cell = cell(1,size(nodelayers,2)-1);
    
    % BEGIN RUNNING MINIBATCHES WITHIN THE EPOCH
    
    % initiate the initial minibatch
    minibatch_start = 1;
    minibatch_end = batchSize; 

    while minibatch_end <= size(train_input,2)
        % set the initial activation matrix (one column for each instance in
        % the minibatch)
        a_train = train_input(:,minibatch_start:minibatch_end);
        
        activation_cell{1} = a_train;
        
        % feed forward
        for l = 1:size(nodelayers,2)-1
            % turn the bias vector into a matrix consistent with the number of
            % instances in the minibatch
            bias_matrix = repmat(bias_cell{l},1,batchSize);

            % compute the weighted input for the next layer
            z = weights_cell{l}*a_train + bias_matrix;
            z_cell{l} = z;
            
            if strcmp(transfer,'softmax') == 1 
                if l == size(nodelayers,2)-1
                    a_train = softmax(z);
                    activation_cell{l+1} = a_train;
                else
                    a_train = logsig(z);
                    activation_cell{l+1} = a_train;
                end
            elseif strcmp(transfer,'sigmoid') == 1
                a_train = logsig(z); % Sigmoid activation function
            elseif strcmp(transfer,'tanh') == 1
                a_train = tansig(z); % Tanh activation function
            elseif strcmp(transfer,'relu') == 1
                a_train = max(0,z); % ReLU activation function
            end
            
            activation_cell{l+1} = a_train;
        end
        
        if strcmp(transfer,'tanh') == 1
            % normalize output layer to 0-1
            a_min = min(min(a_train));
            a_max = max(max(a_train));
            a_train = (a_train-a_min)/(a_max-a_min);
            activation_cell{l+1} = a_train;
        end

        % compute the error at the output layer
        if strcmp(cost,'quad') == 1 % QUADRATIC COST
            if strcmp(transfer,'softmax') == 1     % sigmoid
                delta_L = (a_train - train_target(:,minibatch_start:minibatch_end)) .* (logsig(z) .* (1 - logsig(z)));
            elseif strcmp(transfer,'sigmoid') == 1     % sigmoid
                delta_L = (a_train - train_target(:,minibatch_start:minibatch_end)) .* (logsig(z) .* (1 - logsig(z)));
            elseif strcmp(transfer,'tanh') == 1     % tanh 
                delta_L = (a_train - train_target(:,minibatch_start:minibatch_end)) .* dtansig(z,tansig(z));
            elseif strcmp(transfer,'relu') == 1     % ReLU
                z = max(z,0); % negatives go to zero
                z = z~=0; % all else go to one
                delta_L = (a_train - train_target(:,minibatch_start:minibatch_end)) .* z;
            end
        elseif strcmp(cost,'log') == 1 % LOG LIKELIHOOD COST
            if strcmp(transfer,'softmax') == 1     % sigmoid
                delta_L = (1 ./ a_train) .* (logsig(z) .* (1 - logsig(z)));
            elseif strcmp(transfer,'sigmoid') == 1     % sigmoid
                delta_L = (1 ./ a_train) .* (logsig(z) .* (1 - logsig(z)));
            elseif strcmp(transfer,'tanh') == 1     % tanh 
                delta_L = (1 ./ a_train) .* dtansig(z,tansig(z));
            elseif strcmp(transfer,'relu') == 1     % ReLU
                z = max(z,0); % negatives go to zero
                z = z~=0; % all else go to one
                delta_L = (1 ./ a_train) .* z;
            end
        elseif strcmp(cost,'cross') == 1 % CROSS ENTROPY COST
            if strcmp(transfer,'softmax') == 1     % sigmoid
                delta_L = ((a_train - train_target(:,minibatch_start:minibatch_end))  ./ (a_train+1) .* a_train) .* (logsig(z) .* (1 - logsig(z)));
            elseif strcmp(transfer,'sigmoid') == 1     % sigmoid
                delta_L = ((a_train - train_target(:,minibatch_start:minibatch_end))  ./ (a_train+1) .* a_train) .* (logsig(z) .* (1 - logsig(z)));
            elseif strcmp(transfer,'tanh') == 1     % tanh  
                delta_L = ((a_train - train_target(:,minibatch_start:minibatch_end))  ./ (a_train+1) .* a_train) .* dtansig(z,tansig(z));
            elseif strcmp(transfer,'relu') == 1     % ReLU
                z = max(z,0); % negatives go to zero
                z = z~=0; % all else go to one
                delta_L = ((a_train - train_target(:,minibatch_start:minibatch_end))  ./ (a_train+1) .* a_train) .* z;
            end
        end
        
        % add the errors at the output layer to the errors cell
        errors_cell{l} = delta_L;
        
        % compute the accuracy
        diff = sum(abs(train_target(:,minibatch_start:minibatch_end) - round(a_train)),1);
            % find all intances where all the guesses are correct
        all_correct = diff == 0;
            % add up all the total correct responses
        correct_guess_train = correct_guess_train + sum(all_correct);
        
        % calculate the contribution to COST
            % must calculate the sum of the squared weights for the entire
            % network to add to the cost for L2 Regularization
        sigma_weights_squared = 0;
        for l = 1:size(weights_cell,2)
            sigma_weights_squared = sigma_weights_squared + sum(dot(weights_cell{l},weights_cell{l}));
        end
        
        if strcmp(cost,'quad') == 1 % QUADRATIC COST
            COST_train = COST_train + (norm((train_target(:,minibatch_start:minibatch_end) - a_train)) ^2) + ((lambda/2*batchSize)*sigma_weights_squared);
        elseif strcmp(cost,'log') == 1 % LOG LIKELIHOOD COST
            COST_train = COST_train + sum(-log(a_train))+ ((lambda/2*batchSize)*sigma_weights_squared);
        elseif strcmp(cost,'cross') == 1 % CROSS ENTROPY COST 
            COST_train = COST_train + sum(sum((train_target(:,minibatch_start:minibatch_end) .* log(a_train) + (1 - train_target(:,minibatch_start:minibatch_end)) .* log(1 - a_train)) + ((lambda/2*batchSize)*sigma_weights_squared))); 
        end
        
        % backpropogate the error
        for back_layer = l-1:-1:1
            errors_cell{back_layer} = weights_cell{back_layer+1}' * errors_cell{back_layer+1} .* (logsig(z_cell{back_layer}) .* (1 - logsig(z_cell{back_layer})));
        end

        % update weights and biases with gradient descent
        for layer = l:-1:1
            % APPLY MOMENTUM
                % update velocities
            velocity_cell{layer} = momentum .* velocity_cell{layer} - (errors_cell{layer} * activation_cell{layer}') / batchSize;
                % update weights
            weights_cell{layer} = weights_cell{layer} + velocity_cell{layer};
            
            % update weights
            %weights_cell{layer} = weights_cell{layer} - (eta / batchSize) * (errors_cell{layer} * activation_cell{layer}');
                % with L2 regularization
            weights_cell{layer} = (1 - (eta*lambda/batchSize)) * weights_cell{layer} - (eta / batchSize) * (errors_cell{layer} * activation_cell{layer}');
            
            % update biases
            bias_cell{layer} = bias_cell{layer} - (eta / batchSize) * sum(errors_cell{layer},2);
        end
        
        % increment the mini batch counters
        minibatch_start = minibatch_start + batchSize;
        minibatch_end = minibatch_end + batchSize;
    end
    
    
    % run the test data through the updated network
    a_test = test_input; 
    % feed forward
    for l = 1:size(nodelayers,2)-1
        % turn the bias vector into a matrix consistent with the number of
        % instances in the minibatch
        bias_matrix = repmat(bias_cell{l},1,size(test_input,2));

        % compute the weighted input for the next layer
        z = weights_cell{l}*a_test + bias_matrix;
        
        if strcmp(transfer,'softmax') == 1   
            if l == size(nodelayers,2)-1
                a_test = softmax(z);
            else
                a_test = logsig(z);
            end
        elseif strcmp(transfer,'sigmoid') == 1 % sigmoid
            a_test = logsig(z); 
        elseif strcmp(transfer,'tanh') == 1     % tanh 
            a_test = tansig(z); % Tanh activation function
        elseif strcmp(transfer,'relu') == 1     % ReLU
            a_test = max(0,z);
        end
    end
    
    if strcmp(transfer,'tanh') == 1     % tanh 
        % normalize output layer to 0-1
        a_min = min(min(a_test));
        a_max = max(max(a_test));
        a_test = (a_test-a_min)/(a_max-a_min);
    end
    
    
    % run the validation data through the updated network
    a_validate = validate_input; 
    % feed forward
    for l = 1:size(nodelayers,2)-1
        % turn the bias vector into a matrix consistent with the number of
        % instances in the minibatch
        bias_matrix = repmat(bias_cell{l},1,size(validate_input,2));

        % compute the weighted input for the next layer
        z = weights_cell{l}*a_validate + bias_matrix;
        
        if strcmp(transfer,'softmax') == 1 
            if l == size(nodelayers,2)-1
                a_validate = softmax(z);
            else
                a_validate = logsig(z);
            end
        elseif strcmp(transfer,'sigmoid') == 1 % sigmoid
            a_validate = logsig(z);
        elseif strcmp(transfer,'tanh') == 1     % tanh 
            a_validate = tansig(z); % Tanh activation function
        elseif strcmp(transfer,'relu') == 1     % ReLU
            a_validate = max(0,z); % ReLU activation function
        end
    end
    
    if strcmp(transfer,'tanh') == 1     % tanh 
        % normalize output layer to 0-1
        a_min = min(min(a_validate));
        a_max = max(max(a_validate));
        a_validate = (a_validate-a_min)/(a_max-a_min);
    end
    
    % compute the COST on the training set based on incremental samples
    % collected in the minibataches
    if strcmp(cost,'quad') == 1 % QUADRATIC COST
        COST_train = COST_train / (2 * size(train_target,2));
    elseif strcmp(cost,'cross') == 1 % CROSS ENTROPY COST  
        COST_train = COST_train / (size(train_target,2));
    end
    
    % note that log likelihood is omitted because this cost does not
    % require that you divide by x
    
    % calculate COST for the test and train datasets
       % must calculate the sum of the squared weights for the entire
       % network for L2 Reguluarization
    sigma_weights_squared = 0;
    for l = 1:size(weights_cell,2)
        sigma_weights_squared = sigma_weights_squared + sum(dot(weights_cell{l},weights_cell{l}));
    end
    
    if strcmp(cost,'quad') == 1 % QUADRATIC COST
        COST_test = (norm((test_target - a_test)) ^2) + ((lambda/2*size(test_target,2))*sigma_weights_squared) / (2 * size(test_target,2));
    elseif strcmp(cost,'log') == 1 % LOG LIKELIHOOD COST
        COST_test = sum(-log(a_test))+ ((lambda/2*size(test_target,2))*sigma_weights_squared);
    elseif strcmp(cost,'cross') == 1 %CROSS ENTROPY COST 
        COST_test = sum(sum(test_target .* log(a_test) + (1 - test_target) .* log(1 - a_test) / size(test_input,2)) + ((lambda/2*size(test_target,2))*sigma_weights_squared));
    end
    
    % check to see if the COST on the validation data went up
    if exist('COST_validate') == 0
        COST_validate = 0;
    end
    COST_previous_epoch = COST_validate;
    
    % set the new cost for the epoch
    if strcmp(cost,'quad') == 1 % QUADRATIC COST
        COST_validate = (norm((validate_target - a_validate)) ^2) + ((lambda/2*size(validate_target,2))*sigma_weights_squared) / (2 * size(validate_target,2));
    elseif strcmp(cost,'log') == 1 % LOG LIKELIHOOD COST
        COST_validate = sum(-log(a_validate)) + ((lambda/2*size(validate_target,2))*sigma_weights_squared);
    elseif strcmp(cost,'cross') == 1 %CROSS ENTROPY COST 
        COST_validate = sum(sum(validate_target .* log(a_validate) + (1 - validate_target) .* log(1 - a_validate) / size(validate_input,2)) + ((lambda/2*size(validate_target,2))*sigma_weights_squared));
    end
    
    if COST_validate > COST_previous_epoch
        COST_validate_increase_counter = COST_validate_increase_counter + 1;
    else
        COST_validate_increase_counter = 0;
    end  
    
    % calculate the correct guesses for the test input
    diff = sum(abs(test_target - round(a_test)),1);
            % find all intances where all the guesses are correct
    all_correct = diff == 0;
            % add up all the total correct responses
    correct_guess_test =  sum(all_correct);
    
    % calculate the correct guesses for the validation input
    diff = sum(abs(validate_target - round(a_validate)),1);
            % find all intances where all the guesses are correct
    all_correct = diff == 0;
            % add up all the total correct responses
    correct_guess_validate =  sum(all_correct);
    
    % add the results as a new row to the appropriate matrices
    train_results = [train_results;[epoch correct_guess_train/size(train_target,2) COST_train]];
    test_results = [test_results; [epoch correct_guess_test/size(test_target,2) COST_test]];
    validate_results = [validate_results; [epoch correct_guess_validate/size(validate_target,2) COST_validate]];
        
    % write the results
    fprintf('%i  | %f | %i/%i | %f     ||%f | %i/%i | %f  || %f | %i/%i | %f\n',epoch...
        , COST_train,    correct_guess_train,    size(train_target,2),    correct_guess_train/size(train_target,2)...
        , COST_test,     correct_guess_test,     size(test_target,2),     correct_guess_test/size(test_target,2)...
        , COST_validate, correct_guess_validate, size(validate_target,2), correct_guess_validate/size(validate_target,2))
    
    % break if the cost for the validation data goes up for 5 consecutive
    % epochs
    if COST_validate_increase_counter == 3
        fprintf('Early stopping due to consecutive increases in the cost on the validation set.\n')
        break
    end
    
    % increase the epoch counter
    epoch = epoch + 1;
end 