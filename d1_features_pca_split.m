% d1_features_pca_split.m — PCA on TRAIN only, then transform TEST (and VAL if exists)
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

% --- Load split indices ---
S = load('splits/splits.mat'); 

% Some setups have only train/test; handle both cases
trainIdx = S.trainIdx;
if isfield(S,'valIdx')
    valIdx = S.valIdx;
else
    valIdx = []; % optional validation
end
if isfield(S,'testIdx')
    testIdx = S.testIdx;
else
    error('No testIdx found in splits.mat');
end

% --- Load features ---
R = load('features/raw/features_raw.mat'); 
X = double(R.X_raw); 
y = double(R.y); 
fileNames = R.fileNames;

% --- Standardize based on TRAIN only ---
mu = mean(X(trainIdx,:), 1);
sigma = std(X(trainIdx,:), 0, 1);
sigma(sigma < eps) = 1;
Z = (X - mu) ./ sigma;

% --- Fit PCA on TRAIN subset ---
[coeff, scoreTrain, latent, ~, explained] = pca(Z(trainIdx,:));

% Choose number of components (95% variance)
targetVar = 0.95;
cumvar = cumsum(explained);
k = find(cumvar >= targetVar*100, 1, 'first');

% --- Transform TRAIN/VAL/TEST using PCA projection ---
Xtr_pca = scoreTrain(:,1:k);

if ~isempty(valIdx)
    Zval = Z(valIdx,:) * coeff;
    Xval_pca = Zval(:,1:k);
else
    Xval_pca = [];
end

Zte = Z(testIdx,:) * coeff;
Xte_pca = Zte(:,1:k);

% --- Save results ---
if ~exist('features/pca','dir'), mkdir('features/pca'); end
save('features/pca/features_pca.mat', ...
    'Xtr_pca','Xval_pca','Xte_pca','y','fileNames', ...
    'trainIdx','valIdx','testIdx', ...
    'coeff','mu','sigma','k','targetVar','explained','latent','-v7.3');

fprintf('✅ PCA done: kept %d comps (%.2f%% var).\n', k, cumvar(k));
fprintf('Train: %d×%d | Test: %d×%d\n', size(Xtr_pca,1),k, size(Xte_pca,1),k);
if ~isempty(Xval_pca)
    fprintf('Validation: %d×%d\n', size(Xval_pca,1),k);
else
    fprintf('Validation: not defined (2-way split)\n');
end