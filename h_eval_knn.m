% h_eval_knn.m — evaluate KNN models 
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

% load data
S = load('splits/splits.mat'); 
testIdx = S.testIdx;
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

% load models and normalization parameters
MR = load('models/knn/modelKNN_raw.mat');
MH = load('models/knn/modelKNN_hog.mat');

modelKNN_raw = MR.modelKNN_raw;
bestRaw = MR.bestRaw;
mu_raw_full = MR.mu_raw_full;
sigma_raw_full = MR.sigma_raw_full;

modelKNN_hog = MH.modelKNN_hog;
bestHog = MH.bestHog;
mu_hog_full = MH.mu_hog_full;
sigma_hog_full = MH.sigma_hog_full;

Xte_raw = double(R.X_raw(testIdx,:)); 
yte = double(R.y(testIdx));
Xte_hog = double(H.X_hog(testIdx,:));

% normalise test data
Xte_raw_norm = (Xte_raw - mu_raw_full) ./ sigma_raw_full;
Xte_hog_norm = (Xte_hog - mu_hog_full) ./ sigma_hog_full;

% predict
tic; 
pred_raw = predict(modelKNN_raw, Xte_raw_norm);  
t_raw = toc/numel(yte);

tic; 
pred_hog = predict(modelKNN_hog, Xte_hog_norm);  
t_hog = toc/numel(yte);

% metrics
acc_raw = mean(pred_raw==yte)*100;
acc_hog = mean(pred_hog==yte)*100;

% Precision, Recall, F1 for class 1
tp_raw = sum((pred_raw==1) & (yte==1));
fp_raw = sum((pred_raw==1) & (yte==0));
fn_raw = sum((pred_raw==0) & (yte==1));
prec_raw = tp_raw/(tp_raw+fp_raw+eps);
rec_raw = tp_raw/(tp_raw+fn_raw+eps);
f1_raw = 2*prec_raw*rec_raw/(prec_raw+rec_raw+eps);

tp_hog = sum((pred_hog==1) & (yte==1));
fp_hog = sum((pred_hog==1) & (yte==0));
fn_hog = sum((pred_hog==0) & (yte==1));
prec_hog = tp_hog/(tp_hog+fp_hog+eps);
rec_hog = tp_hog/(tp_hog+fn_hog+eps);
f1_hog = 2*prec_hog*rec_hog/(prec_hog+rec_hog+eps);

fprintf('\nKNN EVALUATION RESULTS\n');
fprintf('KNN RAW: %.2f%% acc | P=%.3f R=%.3f F1=%.3f | K=%d, %s | %.4fs/sample\n', ...
    acc_raw, prec_raw, rec_raw, f1_raw, bestRaw.K, bestRaw.dist, t_raw);
fprintf('KNN HOG: %.2f%% acc | P=%.3f R=%.3f F1=%.3f | K=%d, %s | %.4fs/sample\n', ...
    acc_hog, prec_hog, rec_hog, f1_hog, bestHog.K, bestHog.dist, t_hog);

% confusion matrices
try
    figure('Position', [100 100 800 350]);
    subplot(1,2,1);
    confusionchart(categorical(yte), categorical(pred_raw));
    title(sprintf('KNN RAW (%.2f%%, K=%d, %s)', acc_raw, bestRaw.K, bestRaw.dist));
    
    subplot(1,2,2);
    confusionchart(categorical(yte), categorical(pred_hog));
    title(sprintf('KNN HOG (%.2f%%, K=%d, %s)', acc_hog, bestHog.K, bestHog.dist));
catch
    warning('Confusionchart unavailable — skipping plots.');
end
