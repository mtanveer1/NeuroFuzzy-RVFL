function   [TrainingTime,TestingTime,TrainingAccuracy,TestingAccuracy]  = Neuro_Fuzzy_RVFL_train(train_x,train_y,test_x,test_y,Alpha,WeightHidden,C,NumFuzzyRule,activation,No_of_class,clus)

%% 0-1 coding for the train target
NClass=No_of_class;
U_dataY_train = 0:1:NClass-1;
% No_of_class=2; %Total number of class (Binary Classification Problem)
dataY_train_temp = zeros(numel(train_y),No_of_class); %constructed a zero matrix of order [(Total no. of samples) times (No of class)]
for i=1:No_of_class %loop runs column wise
    idx = train_y==U_dataY_train(i);
    dataY_train_temp(idx,i)=1;
end

%% 0-1 coding for the test target
U_dataY_test = 0:1:No_of_class-1;
% No_of_class=2; %Total number of class (Classification Problem)
dataY_test_temp = zeros(numel(test_y),No_of_class); %constructed a zero matrix of order [(Total no. of samples) times (No of class)]
for i=1:No_of_class %loop runs column wise
    idx = test_y==U_dataY_test(i);
    dataY_test_temp(idx,i)=1;
end

%% Training starts
std = 1;
tic
Omega=zeros(size(train_x,1),NumFuzzyRule);

F = zeros(size(train_x,1), NumFuzzyRule); %Fuzzy Layer


%% Clustering Methods
cluster=clus;

if cluster==1
[~,center] = kmeans(train_x, NumFuzzyRule);
elseif cluster==2
[center,~] = fcm(train_x,NumFuzzyRule);
else
    Temptrain_x = randperm(length(train_x));
    indices = Temptrain_x(1:NumFuzzyRule);
    center = train_x(indices,:);
end
%%%Calculating fuzzy membership value%%%
for j = 1:size(train_x,1)   
    MF = exp(-(repmat(train_x(j,:), NumFuzzyRule,1) - center).^2/std);
    MF = prod(MF,2); 
    MF = MF/sum(MF);  
    F(j,:) = MF'.*(train_x(j,:)*Alpha); 
    Omega(j,:) = MF;
end

%%%%%%%%%%%%%
F1 = [F,  0.1 * ones(size(F,1),1)];

H = F1 * WeightHidden;  

if activation == 1
    H = sigmoid(H,0,1);
elseif activation == 2
    H = sin(H);
elseif activation == 3
    H = tribas(H);
elseif activation == 4
    H = radbas(H);
elseif activation == 5
    H = tansig(H);
elseif activation == 6
    H = relu(H);
end

H=[H,train_x]; %Direct Link

M=[Omega.*(train_x * Alpha),H]; 

[Nsample,~] = size(train_x);


%%%%%%%%%%%%%%%Finding Output Layer Parameter (Here, beta)%%%%%%%%%%%%%%%%%
if size(M,2)<Nsample
    beta = (M'*M + eye(size(M',1))*(1/C)) \ ( M'  *  dataY_train_temp);
else
    beta = M'*((eye(size(M,1))*(1/C)+M*M') \ dataY_train_temp);
end


PredictedTrainLabel = M * beta; %Test Prediction
TrainingTime = toc;  %Training Time

%Training Accuracy
trainY_temp1 = bsxfun(@minus,PredictedTrainLabel,max(PredictedTrainLabel,[],2));
num = exp(trainY_temp1);
dem = sum(num,2);
prob_scores = bsxfun(@rdivide,num,dem);
[~,indx] = max(prob_scores,[],2);
[~, ind_corrClass] = max(dataY_train_temp,[],2);
TrainingAccuracy = mean(indx == ind_corrClass)*100;

clear F; clear F1; clear MF;

%% Testing starts
tic;
Omega1=zeros(size(test_x,1),NumFuzzyRule);
F= zeros(size(test_x,1), NumFuzzyRule);
for j = 1:size(test_x,1)
    MF = exp(-(repmat(test_x(j,:), NumFuzzyRule,1) - center).^2/std);
    MF = prod(MF,2);
    MF = MF/sum(MF);
    F(j,:) = MF'.*(test_x(j,:)*Alpha);
    Omega1(j,:) = MF;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F1 = [F, .1 * ones(size(F,1),1)]; 

H1 = F1 * WeightHidden; 

if activation == 1
    H1 = sigmoid(H1);
elseif activation == 2
    H1 = sin(H1);
elseif activation == 3
    H1 = tribas(H1);
elseif activation == 4
    H1 = radbas(H1);
elseif activation == 5
    H1 = tansig(H1);
elseif activation == 6
    H1 = relu(H1);
end

H1=[H1,test_x]; %Direct Link

M1=[Omega1.*(test_x * Alpha),H1]; 

PredictedTestLabel = M1 * beta; %Test Prediction
TestingTime = toc; %Testing Time

%Testing Accuracy
testY_temp1 = bsxfun(@minus,PredictedTestLabel,max(PredictedTestLabel,[],2));
num = exp(testY_temp1);
dem = sum(num,2);
prob_scores = bsxfun(@rdivide,num,dem);
[~,indx] = max(prob_scores,[],2);
[~, ind_corrClass] = max(dataY_test_temp,[],2);
TestingAccuracy = mean(indx == ind_corrClass)*100;

clear F; clear F1; clear MF;

end
