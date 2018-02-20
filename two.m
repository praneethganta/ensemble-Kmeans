X_train = importdata('X_train.mat');
y_train = importdata('y_train.mat');
X_test = importdata('X_test.mat');
y_test = importdata('y_test.mat');
%K nearest algorithm
model = fitcknn(X_train,y_train,'NumNeighbors',7);
result_knearest = transpose(predict(model,X_test));
performance = classperf(y_test,result_knearest);
modelPerformance = performance.CorrectRate*100;
fprintf('Percentage Accuracy of k nearnest algorithm: %f\n',modelPerformance);
% SVM
y_test = transpose(y_test);
y_train = transpose(y_train);
y_train_trans(1:10,1:500) = -1;
for i = 1 : size(y_train,2)
    y_train_trans(y_train(i),i) = 1;
end
for i = 1 : size(y_train_trans,1)
    svmMode{i} = fitcsvm(X_train,y_train_trans(i,:),'KernelFunction','polynomial','PolynomialOrder',2);
end
for i = 1 : size(y_train_trans,1)
    y{i} = predict(svmMode{i},X_test);
end

a= zeros(3251,10);
for i = 1 : size(y_train_trans)
    a(:,i) = y{i};
end
result_svm = zeros(3251,1);
for i = 1 : 3251
    result_svm(i) = min((find(a(i,:) == max(a(i,:)))));
end
result_svm = transpose(result_svm);
performance = classperf(result_svm,y_test);

modelPerformance = performance.CorrectRate*100;
fprintf('Percentage Accuracy in svm: %f\n',modelPerformance);

% Neural network

y_train_trans = zeros(25,500);
for i = 1 : size(y_train,2)
    y_train_trans(y_train(i),i) = 1;
end

fb = feedforwardnet(25);
x_train_trans = transpose(X_train);
net = train(fb,x_train_trans,y_train_trans);
x_test_trans = transpose(X_test);
y = net(x_test_trans);

result_neural = zeros(1,3251);
for j = 1 : 3251
    result_neural(j) = (find(y(:,j) == max(y(:,j))));
end
 
performance = classperf(result_neural,y_test);

modelPerformance = performance.CorrectRate*100;
fprintf('Percentage Accuracy in feedforward neural network: %f\n',modelPerformance);
%ensemble
result_combine = [result_knearest; result_neural ;result_svm];
result_ensemble = mode(result_combine,1);
performance = classperf(result_ensemble,y_test);
modelPerformance = performance.CorrectRate*100;
fprintf('Percentage Accuracy in Ensemble: %f\n',modelPerformance);