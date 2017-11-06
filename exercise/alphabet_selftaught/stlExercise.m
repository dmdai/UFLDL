%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%
%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.
clear all;close all;clc
inputSize  = 16 * 16;
numLabels  = 9;
hiddenSize = 196;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;
epsilon = 0.1;
%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
% mnistData   = loadMNISTImages('mnist/train-images-idx3-ubyte');
% mnistLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
% [mnistData mnistLabels] = loadMNISTImages('train');



% Set Unlabeled Set (All Images)

% Simulate a Labeled and Unlabeled set
% labeledSet   = find(mnistLabels >= 0 & mnistLabels <= 4);
% unlabeledSet = find(mnistLabels >= 5);

% temp = randperm(length(unlabeledSet));
% unlabeledSet = temp(1:10000);
% clear temp


% numTrain = round(numel(labeledSet)/2);
% temp = randperm(length(labeledSet));
% trainSet = temp(1:2000);
% testSet = temp(end-1999:end);
% trainSet = labeledSet(1:numTrain);
% testSet  = labeledSet(end-numTrain+1:end);
% testSet  = labeledSet(numTrain+1:end);

% unlabeledData = mnistData(:, unlabeledSet)/255;
% unlabeledData = bsxfun(@minus, unlabeledData, mean(unlabeledData, 2));

% trainData   = mnistData(:, trainSet)/255;
% meanData = mean(trainData, 2);
% trainData = bsxfun(@minus, trainData, meanData);
% trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

% testData   = mnistData(:, testSet)/255;
% testData = bsxfun(@minus, testData, meanData);
% testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5

load('train.mat');
load('test.mat');

unlabeledData = train_data(:, 1:500);
unlabeledData = bsxfun(@minus, unlabeledData, mean(unlabeledData, 2));

trainData = train_data(:, 501:end);
meanData = mean(trainData, 2);
trainData = bsxfun(@minus, trainData, meanData);
trainLabels = train_labels(501:end);

testData = test_data;
testData = bsxfun(@minus, testData, meanData);
testLabels = test_labels;

% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));

%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

%% ----------------- YOUR CODE HERE ----------------------
%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages

opttheta = theta; 
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 200;
options.display = 'on';
[opttheta cost] = minFunc(@(t) sparseAutoencoderCost(t, ...
    inputSize, hiddenSize, ...
    lambda, sparsityParam, ...
    beta, unlabeledData), ...
    theta, options);

%% -----------------------------------------------------
                          
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
figure('name', 'selftaught_features.jpg');
display_network(W1');
print -djpeg selftaught_features.jpg

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  
%  You need to complete the code in feedForwardAutoencoder.m so that the 
%  following command will extract features from the data.

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);

%%======================================================================
%% STEP 4: Train the softmax classifier

softmaxModel = struct;  
%% ----------------- YOUR CODE HERE ----------------------
%  Use softmaxTrain.m from the previous exercise to train a multi-class
%  classifier. 

%  Use lambda = 1e-4 for the weight regularization for softmax

% You need to compute softmaxModel using softmaxTrain on trainFeatures and
% trainLabels

lambda = 1e-4;
numClasses = length(unique(trainLabels));

softmaxModel = softmaxTrain(hiddenSize, numClasses, lambda, ...
    trainFeatures, trainLabels, options);





%% -----------------------------------------------------


%%======================================================================
%% STEP 5: Testing 

%% ----------------- YOUR CODE HERE ----------------------
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel

pred = softmaxPredict(softmaxModel, testFeatures);













%% -----------------------------------------------------

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

% (note that we shift the labels by 1, so that digit 0 now corresponds to
%  label 1)
%
% Accuracy is the proportion of correctly classified images
% The results for our implementation was:
%
% Accuracy: 98.3%
%
% 
