%{
% XOR DATA
% read the data from csv
M = csvread('xor.csv');
% define the inputs
inputs = M(:,1:2)';
% define the targets
targets = M(:,3)';
%}

%{
% IRIS DATA
% read the data from csv
M = csvread('iris.csv');
% define the inputs
inputs = M(:,1:4)';
% define the targets
targets = M(:,5:7)';
%}

%%{
% MNIST DATA
% load the data
load('mnistTrn.mat');
% define the inputs
inputs = trn;
% define the targets
targets = trnAns;
%%}

% define the stucture of the network
nodelayers = [size(inputs,1) 3 2 size(targets,1)];

% define the hyperparameters
numEpochs = 20;
batchSize = 1;
eta = .1;
transfer = 'sigmoid'; % must be one of 'relu', 'tanh', 'sigmoid', 'softmax'
cost = 'quad'; % must be one of 'quad', 'log', 'cross'
momentum = .3;
lambda = 5;

% define the split for validation and test data
split = [80,10,10];

% execute the function
%net1(inputs,targets,nodelayers,numEpochs,batchSize,eta);
[weights_cell,bias_cell,train_results,test_results,validate_results] = net2(inputs,targets,nodelayers,numEpochs,batchSize,eta,lambda,momentum,transfer,cost,split);