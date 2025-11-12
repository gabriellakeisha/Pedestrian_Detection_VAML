%% i_train_svm_rbf.m — RBF-SVM using MATLAB's auto KernelScale
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

if ~exist('models/svm','dir'), mkdir('models/svm'); end

% load data
S = load('splits/splits.mat');
trainIdx = S.trainIdx;
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');
P

Xtr_raw = double(R.X_raw(trainIdx,:));
ytr = double(R.y(trainIdx));
Xtr_hog = double(H.X_hog(trainIdx,:));

fprintf('Training: %d samples | RAW: %d dims | HOG: %d dims\n', ...
    numel(ytr), size(Xtr_raw,2), size(Xtr_hog,2));

% validation split
Ntr = numel(ytr);
p = randperm(Ntr);
Nval = round(0.2*Ntr);
valSel = p(1:Nval);
subSel = p(Nval+1:end);

Xsub_raw = Xtr_raw(subSel,:); ysub = ytr(subSel);
Xval_raw = Xtr_raw(valSel,:); yval = ytr(valSel);
Xsub_hog = Xtr_hog(subSel,:);
Xval_hog = Xtr_hog(valSel,:);

% normalisation
mu_raw = mean(Xsub_raw); sigma_raw = std(Xsub_raw); sigma_raw(sigma_raw<eps)=1;
mu_hog = mean(Xsub_hog); sigma_hog = std(Xsub_hog); sigma_hog(sigma_hog<eps)=1;

Xsub_raw_norm = (Xsub_raw - mu_raw) ./ sigma_raw;
Xval_raw_norm = (Xval_raw - mu_raw) ./ sigma_raw;
Xsub_hog_norm = (Xsub_hog - mu_hog) ./ sigma_hog;
Xval_hog_norm = (Xval_hog - mu_hog) ./ sigma_hog;

fprintf('\nRBF-SVM with AUTO KernelScale\n');

Cset = [0.1 1 10 100 1000];

% full images
fprintf('\nRAW features:\n');
bestRaw.acc = -inf;
for C = Cset
    fprintf('  C=%-5g ... ', C);
    tic;
    
    mdl = fitcsvm(Xsub_raw_norm, ysub, ...
        'KernelFunction', 'rbf', ...
        'KernelScale', 'auto', ...    
        'BoxConstraint', C, ...
        'ClassNames', [0 1], ...
        'Standardize', false);
    
    pred = predict(mdl, Xval_raw_norm);
    acc = mean(pred==yval)*100;
    actual_ks = mdl.KernelParameters.Scale; 
    
    fprintf('%.2f%% (ks=%.1f, %.1fs)\n', acc, actual_ks, toc);
    
    if acc > bestRaw.acc
        bestRaw = struct('acc',acc,'C',C,'ks',actual_ks);
        fprintf('NEW BEST!\n');
    end
end

% HOG
fprintf('\nHOG features:\n');
bestHog.acc = -inf;
for C = Cset
    fprintf('  C=%-5g ... ', C);
    tic;
    
    mdl = fitcsvm(Xsub_hog_norm, ysub, ...
        'KernelFunction', 'rbf', ...
        'KernelScale', 'auto', ...
        'BoxConstraint', C, ...
        'ClassNames', [0 1], ...
        'Standardize', false);
    
    pred = predict(mdl, Xval_hog_norm);
    acc = mean(pred==yval)*100;
    actual_ks = mdl.KernelParameters.Scale;
    
    fprintf('%.2f%% (ks=%.1f, %.1fs)\n', acc, actual_ks, toc);
    
    if acc > bestHog.acc
        bestHog = struct('acc',acc,'C',C,'ks',actual_ks);
        fprintf('NEW BEST!\n');
    end
end

fprintf('\nBEST RESULTS\n');
fprintf('RAW: C=%g, ks=%.1f → %.2f%% validation\n', bestRaw.C, bestRaw.ks, bestRaw.acc);
fprintf('HOG: C=%g, ks=%.1f → %.2f%% validation\n', bestHog.C, bestHog.ks, bestHog.acc);

%% final
mu_raw_full = mean(Xtr_raw); sigma_raw_full = std(Xtr_raw); sigma_raw_full(sigma_raw_full<eps)=1;
mu_hog_full = mean(Xtr_hog); sigma_hog_full = std(Xtr_hog); sigma_hog_full(sigma_hog_full<eps)=1;

Xtr_raw_norm = (Xtr_raw - mu_raw_full) ./ sigma_raw_full;
Xtr_hog_norm = (Xtr_hog - mu_hog_full) ./ sigma_hog_full;

fprintf('\nTraining final models...\n');
modelSVM_raw = fitcsvm(Xtr_raw_norm, ytr, ...
    'KernelFunction','rbf', 'KernelScale',bestRaw.ks, ...
    'BoxConstraint',bestRaw.C, 'ClassNames',[0 1], 'Standardize',false);

modelSVM_hog = fitcsvm(Xtr_hog_norm, ytr, ...
    'KernelFunction','rbf', 'KernelScale',bestHog.ks, ...
    'BoxConstraint',bestHog.C, 'ClassNames',[0 1], 'Standardize',false);

save('models/svm/modelSVM_rbf_raw.mat', 'modelSVM_raw', 'bestRaw', ...
     'mu_raw_full', 'sigma_raw_full', '-v7.3');
save('models/svm/modelSVM_rbf_hog.mat', 'modelSVM_hog', 'bestHog', ...
     'mu_hog_full', 'sigma_hog_full', '-v7.3');

fprintf('   RAW: C=%g, KernelScale=%.1f\n', bestRaw.C, bestRaw.ks);
fprintf('   HOG: C=%g, KernelScale=%.1f\n', bestHog.C, bestHog.ks);
