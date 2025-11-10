% f_eval_nn.m — evaluate NN
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

S = load('splits/splits.mat');
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

load('models/nn/modelNN_raw.mat');
load('models/nn/modelNN_hog.mat');

testIdx = S.testIdx;
Xte_raw = double(R.X_raw(testIdx,:)); 
yte = double(R.y(testIdx));
Xte_hog = double(H.X_hog(testIdx,:));

% normalisation
Xte_raw_norm = (Xte_raw - modelNN_raw.mu) ./ modelNN_raw.sigma;
Xte_hog_norm = (Xte_hog - modelNN_hog.mu) ./ modelNN_hog.sigma;

% full images evaluation
pred_raw = zeros(numel(yte),1);
tic
for i = 1:numel(pred_raw)
    pred_raw(i) = NNTesting(Xte_raw_norm(i,:), modelNN_raw);
end
t_raw = toc/numel(pred_raw);
acc_raw = mean(pred_raw == yte)*100;

% HOG evaluation
pred_hog = zeros(numel(yte),1);
tic
for i = 1:numel(pred_hog)
    pred_hog(i) = NNTesting(Xte_hog_norm(i,:), modelNN_hog);
end
t_hog = toc/numel(pred_hog);
acc_hog = mean(pred_hog == yte)*100;

% metrics
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

fprintf(['\nNN (NORMALIZED) EVALUATION\n']);
fprintf('NN-RAW (norm): %.2f%% | P=%.3f R=%.3f F1=%.3f | %.4fs/sample\n', ...
    acc_raw, prec_raw, rec_raw, f1_raw, t_raw);
fprintf('NN-HOG (norm): %.2f%% | P=%.3f R=%.3f F1=%.3f | %.4fs/sample\n', ...
    acc_hog, prec_hog, rec_hog, f1_hog, t_hog);

try
    figure('Position', [100 100 800 350]);
    subplot(1,2,1);
    confusionchart(categorical(yte), categorical(pred_raw));
    title(sprintf('NN-RAW (%.2f%%)', acc_raw));
    
    subplot(1,2,2);
    confusionchart(categorical(yte), categorical(pred_hog));
    title(sprintf('NN-HOG (%.2f%%)', acc_hog));
catch
    warning('Confusionchart unavailable');
end
