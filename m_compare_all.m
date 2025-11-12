%% m_compare_all.m - Compare all models (NN, KNN, SVM)
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

% Load test data
S = load('splits/splits.mat'); 
testIdx = S.testIdx;
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

Xte_raw = double(R.X_raw(testIdx,:)); 
yte = double(R.y(testIdx));
Xte_hog = double(H.X_hog(testIdx,:));

fprintf('   PEDESTRIAN DETECTION - MODEL COMPARISON\n');
fprintf('Test set: %d samples (Pos: %d, Neg: %d)\n\n', ...
    numel(yte), sum(yte==1), sum(yte==0));

results = struct('name',{},'acc',{},'prec',{},'rec',{},'f1',{},'time',{});
idx = 1;

%% 1. NN
fprintf('Evaluating NN Models\n');

% NN-RAW
try
    load('models/nn/modelNN_raw.mat');
    
    % Normalize test data using training statistics
    Xte_raw_norm = (Xte_raw - modelNN_raw.mu) ./ modelNN_raw.sigma;
    
    pred_raw = zeros(numel(yte),1);
    tic;
    for i = 1:numel(pred_raw)
        pred_raw(i) = NNTesting(Xte_raw_norm(i,:), modelNN_raw);
    end
    t = toc/numel(yte);
    
    acc = mean(pred_raw==yte)*100;
    tp = sum((pred_raw==1) & (yte==1));
    fp = sum((pred_raw==1) & (yte==0));
    fn = sum((pred_raw==0) & (yte==1));
    prec = tp/(tp+fp+eps);
    rec = tp/(tp+fn+eps);
    f1 = 2*prec*rec/(prec+rec+eps);
    
    results(idx) = struct('name','NN-RAW','acc',acc,'prec',prec,'rec',rec,'f1',f1,'time',t);
    fprintf('NN-RAW: %.2f%%\n', acc);
    idx = idx + 1;
catch ME
    fprintf('NN-RAW failed: %s\n', ME.message);
end

% NN-HOG
try
    load('models/nn/modelNN_hog.mat');
    
    % Normalize test data using training statistics
    Xte_hog_norm = (Xte_hog - modelNN_hog.mu) ./ modelNN_hog.sigma;
    
    pred_hog = zeros(numel(yte),1);
    tic;
    for i = 1:numel(pred_hog)
        pred_hog(i) = NNTesting(Xte_hog_norm(i,:), modelNN_hog);
    end
    t = toc/numel(yte);
    
    acc = mean(pred_hog==yte)*100;
    tp = sum((pred_hog==1) & (yte==1));
    fp = sum((pred_hog==1) & (yte==0));
    fn = sum((pred_hog==0) & (yte==1));
    prec = tp/(tp+fp+eps);
    rec = tp/(tp+fn+eps);
    f1 = 2*prec*rec/(prec+rec+eps);
    
    results(idx) = struct('name','NN-HOG','acc',acc,'prec',prec,'rec',rec,'f1',f1,'time',t);
    fprintf('NN-HOG: %.2f%%\n', acc);
    idx = idx + 1;
catch ME
    fprintf('NN-HOG failed: %s\n', ME.message);
end

%% 2. K-NN
fprintf('\nEvaluating KNN Models\n');
try
    MR = load('models/knn/modelKNN_raw.mat');
    
    % Check if normalisation params exist
    if isfield(MR, 'mu_raw_full')
        Xte_raw_norm = (Xte_raw - MR.mu_raw_full) ./ MR.sigma_raw_full;
        norm_str = ' (normalized)';
    else
        Xte_raw_norm = Xte_raw;
        norm_str = ' (NO NORM!)';
    end
    
    tic;
    pred_raw = predict(MR.modelKNN_raw, Xte_raw_norm);
    t = toc/numel(yte);
    
    acc = mean(pred_raw==yte)*100;
    tp = sum((pred_raw==1) & (yte==1));
    fp = sum((pred_raw==1) & (yte==0));
    fn = sum((pred_raw==0) & (yte==1));
    prec = tp/(tp+fp+eps);
    rec = tp/(tp+fn+eps);
    f1 = 2*prec*rec/(prec+rec+eps);
    
    results(idx) = struct('name',sprintf('KNN-RAW (K=%d)%s',MR.bestRaw.K,norm_str),'acc',acc,'prec',prec,'rec',rec,'f1',f1,'time',t);
    fprintf('KNN-RAW: %.2f%% (K=%d)%s\n', acc, MR.bestRaw.K, norm_str);
    idx = idx + 1;
catch ME
    fprintf('KNN-RAW failed: %s\n', ME.message);
end

try
    MH = load('models/knn/modelKNN_hog.mat');
    
    if isfield(MH, 'mu_hog_full')
        Xte_hog_norm = (Xte_hog - MH.mu_hog_full) ./ MH.sigma_hog_full;
        norm_str = ' (normalized)';
    else
        Xte_hog_norm = Xte_hog;
        norm_str = ' (NO NORM!)';
    end
    
    tic;
    pred_hog = predict(MH.modelKNN_hog, Xte_hog_norm);
    t = toc/numel(yte);
    
    acc = mean(pred_hog==yte)*100;
    tp = sum((pred_hog==1) & (yte==1));
    fp = sum((pred_hog==1) & (yte==0));
    fn = sum((pred_hog==0) & (yte==1));
    prec = tp/(tp+fp+eps);
    rec = tp/(tp+fn+eps);
    f1 = 2*prec*rec/(prec+rec+eps);
    
    results(idx) = struct('name',sprintf('KNN-HOG (K=%d)%s',MH.bestHog.K,norm_str),'acc',acc,'prec',prec,'rec',rec,'f1',f1,'time',t);
    fprintf('KNN-HOG: %.2f%% (K=%d)%s\n', acc, MH.bestHog.K, norm_str);
    idx = idx + 1;
catch ME
    fprintf('KNN-HOG failed: %s\n', ME.message);
end

%% 3. SVM RBF
fprintf('\nEvaluating SVM-RBF Models\n');
try
    MR = load('models/svm/modelSVM_rbf_raw.mat');
    Xte_raw_norm = (Xte_raw - MR.mu_raw_full) ./ MR.sigma_raw_full;
    
    tic;
    pred_raw = predict(MR.modelSVM_raw, Xte_raw_norm);
    t = toc/numel(yte);
    
    acc = mean(pred_raw==yte)*100;
    tp = sum((pred_raw==1) & (yte==1));
    fp = sum((pred_raw==1) & (yte==0));
    fn = sum((pred_raw==0) & (yte==1));
    prec = tp/(tp+fp+eps);
    rec = tp/(tp+fn+eps);
    f1 = 2*prec*rec/(prec+rec+eps);
    
    results(idx) = struct('name',sprintf('SVM-RBF-RAW (C=%g,ks=%g)',MR.bestRaw.C,MR.bestRaw.ks),'acc',acc,'prec',prec,'rec',rec,'f1',f1,'time',t);
    fprintf('SVM-RBF-RAW: %.2f%%\n', acc);
    idx = idx + 1;
catch ME
    fprintf('SVM-RBF-RAW failed: %s\n', ME.message);
end

try
    MH = load('models/svm/modelSVM_rbf_hog.mat');
    Xte_hog_norm = (Xte_hog - MH.mu_hog_full) ./ MH.sigma_hog_full;
    
    tic;
    pred_hog = predict(MH.modelSVM_hog, Xte_hog_norm);
    t = toc/numel(yte);
    
    acc = mean(pred_hog==yte)*100;
    tp = sum((pred_hog==1) & (yte==1));
    fp = sum((pred_hog==1) & (yte==0));
    fn = sum((pred_hog==0) & (yte==1));
    prec = tp/(tp+fp+eps);
    rec = tp/(tp+fn+eps);
    f1 = 2*prec*rec/(prec+rec+eps);
    
    results(idx) = struct('name',sprintf('SVM-RBF-HOG (C=%g,ks=%g)',MH.bestHog.C,MH.bestHog.ks),'acc',acc,'prec',prec,'rec',rec,'f1',f1,'time',t);
    fprintf('SVM-RBF-HOG: %.2f%%\n', acc);
    idx = idx + 1;
catch ME
    fprintf('SVM-RBF-HOG failed: %s\n', ME.message);
end

%% 4. SVM Linear
fprintf('\nEvaluating SVM-Linear Models\n');
try
    MR = load('models/svm/modelSVM_linear_raw.mat');
    Xte_raw_norm = (Xte_raw - MR.mu_raw_full) ./ MR.sigma_raw_full;
    
    tic;
    pred_raw = predict(MR.modelSVM_linear_raw, Xte_raw_norm);
    t = toc/numel(yte);
    
    acc = mean(pred_raw==yte)*100;
    tp = sum((pred_raw==1) & (yte==1));
    fp = sum((pred_raw==1) & (yte==0));
    fn = sum((pred_raw==0) & (yte==1));
    prec = tp/(tp+fp+eps);
    rec = tp/(tp+fn+eps);
    f1 = 2*prec*rec/(prec+rec+eps);
    
    results(idx) = struct('name',sprintf('SVM-Linear-RAW (C=%g)',MR.bestRaw.C),'acc',acc,'prec',prec,'rec',rec,'f1',f1,'time',t);
    fprintf('SVM-Linear-RAW: %.2f%%\n', acc);
    idx = idx + 1;
catch ME
    fprintf('SVM-Linear-RAW failed: %s\n', ME.message);
end

try
    MH = load('models/svm/modelSVM_linear_hog.mat');
    Xte_hog_norm = (Xte_hog - MH.mu_hog_full) ./ MH.sigma_hog_full;
    
    tic;
    pred_hog = predict(MH.modelSVM_linear_hog, Xte_hog_norm);
    t = toc/numel(yte);
    
    acc = mean(pred_hog==yte)*100;
    tp = sum((pred_hog==1) & (yte==1));
    fp = sum((pred_hog==1) & (yte==0));
    fn = sum((pred_hog==0) & (yte==1));
    prec = tp/(tp+fp+eps);
    rec = tp/(tp+fn+eps);
    f1 = 2*prec*rec/(prec+rec+eps);
    
    results(idx) = struct('name',sprintf('SVM-Linear-HOG (C=%g)',MH.bestHog.C),'acc',acc,'prec',prec,'rec',rec,'f1',f1,'time',t);
    fprintf('SVM-Linear-HOG: %.2f%%\n', acc);
    idx = idx + 1;
catch ME
    fprintf('SVM-Linear-HOG failed: %s\n', ME.message);
end

%% Final Results
fprintf('FINAL RESULTS\n');
fprintf('%-30s | %7s | %5s | %5s | %5s | %10s\n', ...
    'Model', 'Acc', 'Prec', 'Rec', 'F1', 'Time(s)');
fprintf('%s\n', repmat('-', 85, 1));

for i = 1:numel(results)
    fprintf('%-30s | %6.2f%% | %.3f | %.3f | %.3f | %.6f\n', ...
        results(i).name, results(i).acc, results(i).prec, ...
        results(i).rec, results(i).f1, results(i).time);
end

% Find best models
fprintf('\n%s\n', repmat('=', 85, 1));
[best_acc, best_idx] = max([results.acc]);
[best_f1, bestf1_idx] = max([results.f1]);
[best_speed, fastest_idx] = min([results.time]);

fprintf('BEST ACCURACY: %s (%.2f%%)\n', results(best_idx).name, best_acc);
fprintf('BEST F1-SCORE: %s (%.3f)\n', results(bestf1_idx).name, best_f1);
fprintf('FASTEST MODEL: %s (%.6fs)\n', results(fastest_idx).name, best_speed);
fprintf('%s\n', repmat('=', 85, 1));

