%% j_eval_svm_rbf.m — SVM-RBF model evaluation
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

% load test data
S = load('splits/splits.mat'); 
testIdx = S.testIdx;
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

% load models
MR = load('models/svm/modelSVM_rbf_raw.mat');
MH = load('models/svm/modelSVM_rbf_hog.mat');

Xte_raw = double(R.X_raw(testIdx,:)); 
yte = double(R.y(testIdx));
Xte_hog = double(H.X_hog(testIdx,:));

% normalize test data
Xte_raw_norm = (Xte_raw - MR.mu_raw_full) ./ MR.sigma_raw_full;
Xte_hog_norm = (Xte_hog - MH.mu_hog_full) ./ MH.sigma_hog_full;

% predict
tic; 
[pred_raw, score_raw] = predict(MR.modelSVM_raw, Xte_raw_norm); 
t_raw = toc/numel(yte);

tic; 
[pred_hog, score_hog] = predict(MH.modelSVM_hog, Xte_hog_norm); 
t_hog = toc/numel(yte);

% metrics
acc_raw = mean(pred_raw==yte)*100;
tp_raw = sum((pred_raw==1) & (yte==1));
fp_raw = sum((pred_raw==1) & (yte==0));
fn_raw = sum((pred_raw==0) & (yte==1));
prec_raw = tp_raw/(tp_raw+fp_raw+eps);
rec_raw = tp_raw/(tp_raw+fn_raw+eps);
f1_raw = 2*prec_raw*rec_raw/(prec_raw+rec_raw+eps);

acc_hog = mean(pred_hog==yte)*100;
tp_hog = sum((pred_hog==1) & (yte==1));
fp_hog = sum((pred_hog==1) & (yte==0));
fn_hog = sum((pred_hog==0) & (yte==1));
prec_hog = tp_hog/(tp_hog+fp_hog+eps);
rec_hog = tp_hog/(tp_hog+fn_hog+eps);
f1_hog = 2*prec_hog*rec_hog/(prec_hog+rec_hog+eps);

fprintf('\nOPTIMIZED RBF-SVM RESULTS\n');
fprintf('SVM-RBF-RAW: %.2f%% | P=%.3f R=%.3f F1=%.3f | C=%g, ks=%g | %.4fs\n', ...
    acc_raw, prec_raw, rec_raw, f1_raw, MR.bestRaw.C, MR.bestRaw.ks, t_raw);
fprintf('SVM-RBF-HOG: %.2f%% | P=%.3f R=%.3f F1=%.3f | C=%g, ks=%g | %.4fs\n', ...
    acc_hog, prec_hog, rec_hog, f1_hog, MH.bestHog.C, MH.bestHog.ks, t_hog);

% confusion matrices
figure('Position', [100 100 800 350]);
subplot(1,2,1);
confusionchart(categorical(yte), categorical(pred_raw));
title(sprintf('SVM-RBF-RAW (%.2f%%, C=%g, ks=%g)', acc_raw, MR.bestRaw.C, MR.bestRaw.ks));

subplot(1,2,2);
confusionchart(categorical(yte), categorical(pred_hog));
title(sprintf('SVM-RBF-HOG (%.2f%%, C=%g, ks=%g)', acc_hog, MH.bestHog.C, MH.bestHog.ks));