%% i_train_svm_linear.m — linear SVM
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

if ~exist('models/svm','dir'), mkdir('models/svm'); end

% load data
S = load('splits/splits.mat');
trainIdx = S.trainIdx;
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

Xtr_raw = double(R.X_raw(trainIdx,:));
ytr = double(R.y(trainIdx));
Xtr_hog = double(H.X_hog(trainIdx,:));

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

% Linear SVM start
Cset = [0.01 0.1 1 10 100];

fprintf('\nLINEAR SVM\n');

bestRaw.acc = -inf;
fprintf('\nRAW Features (Linear\n');
for C = Cset
    fprintf('  C=%.3g... ', C);
    tic;
    mdl = fitcsvm(Xsub_raw_norm, ysub, ...
        'KernelFunction','linear', ...
        'BoxConstraint',C, ...
        'ClassNames',[0 1], ...
        'Standardize',false);  
    pred = predict(mdl, Xval_raw_norm);
    acc = mean(pred==yval)*100;
    fprintf('%.2f%% (%.2fs)\n', acc, toc);
    
    if acc > bestRaw.acc
        bestRaw = struct('acc',acc,'C',C);
        fprintf('    *** NEW BEST ***\n');
    end
end

bestHog.acc = -inf;
fprintf('\nHOG Features (Linear)\n');
for C = Cset
    fprintf('  C=%.3g... ', C);
    tic;
    mdl = fitcsvm(Xsub_hog_norm, ysub, ...
        'KernelFunction','linear', ...
        'BoxConstraint',C, ...
        'ClassNames',[0 1], ...
        'Standardize',false);  
    pred = predict(mdl, Xval_hog_norm);
    acc = mean(pred==yval)*100;
    fprintf('%.2f%% (%.2fs)\n', acc, toc);
    
    if acc > bestHog.acc
        bestHog = struct('acc',acc,'C',C);
        fprintf('    *** NEW BEST ***\n');
    end
end

fprintf('\nBEST LINEAR SVM\n');
fprintf('RAW: C=%.3g, val=%.2f%%\n', bestRaw.C, bestRaw.acc);
fprintf('HOG: C=%.3g, val=%.2f%%\n', bestHog.C, bestHog.acc);

% full training set normalization
mu_raw_full = mean(Xtr_raw); sigma_raw_full = std(Xtr_raw); sigma_raw_full(sigma_raw_full<eps)=1;
mu_hog_full = mean(Xtr_hog); sigma_hog_full = std(Xtr_hog); sigma_hog_full(sigma_hog<eps)=1;

Xtr_raw_norm = (Xtr_raw - mu_raw_full) ./ sigma_raw_full;
Xtr_hog_norm = (Xtr_hog - mu_hog_full) ./ sigma_hog_full;

% final models
fprintf('\nTraining final models...\n');
modelSVM_linear_raw = fitcsvm(Xtr_raw_norm, ytr, ...
    'KernelFunction','linear', 'BoxConstraint',bestRaw.C, 'ClassNames',[0 1]);
modelSVM_linear_hog = fitcsvm(Xtr_hog_norm, ytr, ...
    'KernelFunction','linear', 'BoxConstraint',bestHog.C, 'ClassNames',[0 1]);

save('models/svm/modelSVM_linear_raw.mat', 'modelSVM_linear_raw', 'bestRaw', ...
     'mu_raw_full', 'sigma_raw_full', '-v7.3');
save('models/svm/modelSVM_linear_hog.mat', 'modelSVM_linear_hog', 'bestHog', ...
     'mu_hog_full', 'sigma_hog_full', '-v7.3');

fprintf('Linear SVM models saved!\n');
