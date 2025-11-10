%% eval_with_halfhalf_split.m — Evaluate models with 50/50 split
% Quick evaluation of best models using 50/50 split
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

fprintf('\n========================================\n');
fprintf('   EVALUATION WITH 50/50 SPLIT\n');
fprintf('========================================\n\n');

% Load 50/50 split
S = load('splits/splits_50_50.mat');
trainIdx = S.trainIdx;
testIdx = S.testIdx;

fprintf('Training: %d samples (50%%)\n', numel(trainIdx));
fprintf('Testing:  %d samples (50%%)\n\n', numel(testIdx));

% Load features
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

% Prepare training and test sets
Xtr_raw = double(R.X_raw(trainIdx,:));
ytr = double(R.y(trainIdx));
Xtr_hog = double(H.X_hog(trainIdx,:));

Xte_raw = double(R.X_raw(testIdx,:));
yte = double(R.y(testIdx));
Xte_hog = double(H.X_hog(testIdx,:));

fprintf('Testing best models with 50/50 split...\n\n');

results_50 = struct();
idx = 1;

%% Test KNN-HOG
fprintf('--- KNN-HOG ---\n');
try
    % Load best parameters from 70/30 training
    M = load('models/knn/modelKNN_hog.mat');
    K = M.bestHog.K;
    
    % Normalize
    mu = mean(Xtr_hog);
    sigma = std(Xtr_hog);
    sigma(sigma < eps) = 1;
    Xtr_hog_norm = (Xtr_hog - mu) ./ sigma;
    Xte_hog_norm = (Xte_hog - mu) ./ sigma;
    
    % Train
    fprintf('  Training with K=%d... ', K);
    mdl = fitcknn(Xtr_hog_norm, ytr, ...
        'NumNeighbors', K, ...
        'Distance', 'euclidean', ...
        'Standardize', false);
    
    % Test
    tic;
    pred = predict(mdl, Xte_hog_norm);
    t = toc / numel(yte);
    
    % Metrics
    [acc, prec, rec, f1, tp, fp, tn, fn] = calculateMetrics(yte, pred);
    
    results_50(idx).name = sprintf('KNN-HOG (K=%d)', K);
    results_50(idx).acc = acc;
    results_50(idx).prec = prec;
    results_50(idx).rec = rec;
    results_50(idx).f1 = f1;
    results_50(idx).time = t;
    
    fprintf('Acc: %.2f%%\n', acc);
    idx = idx + 1;
catch ME
    fprintf('  ERROR: %s\n', ME.message);
end

%% Test SVM-Linear-HOG
fprintf('--- SVM-Linear-HOG ---\n');
try
    M = load('models/svm/modelSVM_linear_hog.mat');
    C = M.bestHog.C;
    
    mu = mean(Xtr_hog);
    sigma = std(Xtr_hog);
    sigma(sigma < eps) = 1;
    Xtr_hog_norm = (Xtr_hog - mu) ./ sigma;
    Xte_hog_norm = (Xte_hog - mu) ./ sigma;
    
    fprintf('  Training with C=%g... ', C);
    mdl = fitcsvm(Xtr_hog_norm, ytr, ...
        'KernelFunction', 'linear', ...
        'BoxConstraint', C, ...
        'ClassNames', [0 1], ...
        'Standardize', false);
    
    tic;
    pred = predict(mdl, Xte_hog_norm);
    t = toc / numel(yte);
    
    [acc, prec, rec, f1, tp, fp, tn, fn] = calculateMetrics(yte, pred);
    
    results_50(idx).name = sprintf('SVM-Linear-HOG (C=%g)', C);
    results_50(idx).acc = acc;
    results_50(idx).prec = prec;
    results_50(idx).rec = rec;
    results_50(idx).f1 = f1;
    results_50(idx).time = t;
    
    fprintf('Acc: %.2f%%\n', acc);
    idx = idx + 1;
catch ME
    fprintf('  ERROR: %s\n', ME.message);
end

%% Test SVM-RBF-HOG
fprintf('--- SVM-RBF-HOG ---\n');
try
    M = load('models/svm/modelSVM_rbf_hog.mat');
    C = M.bestHog.C;
    ks = M.bestHog.ks;
    
    mu = mean(Xtr_hog);
    sigma = std(Xtr_hog);
    sigma(sigma < eps) = 1;
    Xtr_hog_norm = (Xtr_hog - mu) ./ sigma;
    Xte_hog_norm = (Xte_hog - mu) ./ sigma;
    
    fprintf('  Training with C=%g, ks=%g... ', C, ks);
    mdl = fitcsvm(Xtr_hog_norm, ytr, ...
        'KernelFunction', 'rbf', ...
        'KernelScale', ks, ...
        'BoxConstraint', C, ...
        'ClassNames', [0 1], ...
        'Standardize', false);
    
    tic;
    pred = predict(mdl, Xte_hog_norm);
    t = toc / numel(yte);
    
    [acc, prec, rec, f1, tp, fp, tn, fn] = calculateMetrics(yte, pred);
    
    results_50(idx).name = sprintf('SVM-RBF-HOG (C=%g)', C);
    results_50(idx).acc = acc;
    results_50(idx).prec = prec;
    results_50(idx).rec = rec;
    results_50(idx).f1 = f1;
    results_50(idx).time = t;
    
    fprintf('Acc: %.2f%%\n', acc);
    idx = idx + 1;
catch ME
    fprintf('  ERROR: %s\n', ME.message);
end

%% Display Results
fprintf('\n========================================\n');
fprintf('   50/50 SPLIT RESULTS\n');
fprintf('========================================\n\n');

fprintf('%-25s | %8s | %5s | %5s | %5s\n', ...
    'Model', 'Accuracy', 'Prec', 'Rec', 'F1');
fprintf('%s\n', repmat('-', 70, 1));

for i = 1:numel(results_50)
    fprintf('%-25s | %7.2f%% | %.3f | %.3f | %.3f\n', ...
        results_50(i).name, results_50(i).acc, ...
        results_50(i).prec, results_50(i).rec, results_50(i).f1);
end

fprintf('\n✅ 50/50 evaluation complete!\n');

%% Save results
save('results/results_50_50.mat', 'results_50');
fprintf('   Results saved to: results/results_50_50.mat\n');

%% Helper Function
function [acc, prec, rec, f1, tp, fp, tn, fn] = calculateMetrics(y_true, y_pred)
    tp = sum((y_pred==1) & (y_true==1));
    fp = sum((y_pred==1) & (y_true==0));
    tn = sum((y_pred==0) & (y_true==0));
    fn = sum((y_pred==0) & (y_true==1));
    
    acc = (tp + tn) / (tp + fp + tn + fn) * 100;
    prec = tp / (tp + fp + eps);
    rec = tp / (tp + fn + eps);
    f1 = 2 * prec * rec / (prec + rec + eps);
end