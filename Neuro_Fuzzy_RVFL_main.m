%%
% Please cite the following paper if you are using this code.
% Reference: M. Sajid, A. K. Malik, M. Tanveer, and P. N. Suganthan. “Neuro-Fuzzy Random Vector Functional Link Neural Network for Classification and Regression Problems.” 
% - Revision submitted in IEEE Transactions on Fuzzy Systems.
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The experimental procedures are executed on a computing system possessing MATLAB R2023a software, Intel(R) Xeon(R) Platinum 8260 CPU @ 2.30GHz, 2301 Mhz, 24 Core(s),
% 48 Logical Processor(s) with 256 GB RAM on a Windows-10 operating platform. 
% 
% We have put a demo of the “Neuro_Fuzzy_RVFL” model with the “credit_approval” dataset 
% 
% The following are the parameters set used for the experiment 
% 
% C=0.001; %Regularization parameter
% N=15; %Fuzzy Nodes
% L=203; %Hidden Nodes
% Act=5; %Activation Function

%%
clc;
clear;
warning off all;
format compact;

%% Clustering Methods
cluster=[1,2,3];
clus=2; % Fuzzy C-Means Cluster

% K-Means: clus=1
% Fuzzy C-Means: clus=2
% R-Means: clus=3

%% Data Preparation
split_ratio=0.8; nFolds=5; addpath(genpath('C:\Users\HP\OneDrive - IIT Indore\Desktop\NF-RVFL\Codes'))
temp_data1=load('credit_approval.mat');

temp_data=temp_data1.credit_approval;

[Cls,~,~] = unique(temp_data(:,end));
No_of_class = size(Cls,1);


trainX=temp_data(:,1:end-1); mean_X = mean(trainX,1); std_X = std(trainX);
trainX = bsxfun(@rdivide,trainX-repmat(mean_X,size(trainX,1),1),std_X);
All_Data=[trainX,temp_data(:,end)];

[samples,~]=size(All_Data);
rng('default')
test_start=floor(split_ratio*samples);
training_Data = All_Data(1:test_start-1,:); testing_Data = All_Data(test_start:end,:);
test_x=testing_Data(:,1:end-1); test_y=testing_Data(:,end);
train_x=training_Data(:,1:end-1); train_y=training_Data(:,end);

%% Hyperparameter range
% C=10.^(-5:1:5); %Regularization parameter
% N=5:5:50; %Fuzzy Nodes
% L=3:20:203; %Hidden Nodes
% Act=1:1:6; %Activation Function
C=0.001; %Regularization parameter
N=15; %Fuzzy Nodes
L=203; %Hidden Nodes
Act=5; %Activation Function

NumFuzzyRule=N;
NumHiddenNodes=L;
%% Randomly initializing parameters
Alpha=rand(size(train_x,2),NumFuzzyRule); % Randomly generating coefficients of THEN part of fuzzy rules for the fuzzy layer
WeightHidden=rand(NumFuzzyRule+1,NumHiddenNodes); % Randomly initializing weights and bias connecting fuzzy layer with hidden layer
%% Calling training function
[TrainingTime,TestingTime,TrainingAccuracy,TestingAccuracy] = Neuro_Fuzzy_RVFL_train(train_x,train_y,test_x,test_y,Alpha,WeightHidden,C,N,Act,No_of_class,clus);

fprintf(1, 'Testing Accuracy of NF-RVFL model is: %f\n', TestingAccuracy);

