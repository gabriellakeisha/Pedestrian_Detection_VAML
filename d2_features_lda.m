% d2_features_lda.m — Supervised LDA feature projection using PCA outputs
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

fprintf('LDA Feature Extraction\n');

% --- Load PCA data ---
P = load('features/pca/features_pca.mat');  % output from d1_features_pca_split.m
y = double(P.y);
Xtr_pca = double(P.Xtr_pca);
Xte_pca = double(P.Xte_pca);
Xval_pca = [];
if isfield(P,'Xval_pca')
    Xval_pca = double(P.Xval_pca);
end

% --- Train LDA model on TRAIN set ---
fprintf('Training LDA on PCA-reduced training data...\n');
ldaModel = fitcdiscr(Xtr_pca, y(P.trainIdx), 'DiscrimType','linear');

% --- Project TRAIN / TEST / VAL into LDA space ---
W = ldaModel.Coeffs(1,2).Linear;  % projection vector for binary case
Xtr_lda = Xtr_pca * W;
Xte_lda = Xte_pca * W;
if ~isempty(Xval_pca)
    Xval_lda = Xval_pca * W;
else
    Xval_lda = [];
end

% --- Save ---
if ~exist('features/lda','dir'), mkdir('features/lda'); end
save('features/lda/features_lda.mat', ...
    'Xtr_lda','Xte_lda','Xval_lda','y','ldaModel','W', ...
    'P','-v7.3');

fprintf('LDA done: input %d-D → output %d-D\n', size(Xtr_pca,2), size(W,2));
fprintf('Train: %d×%d | Test: %d×%d\n', size(Xtr_lda,1), size(W,2), size(Xte_lda,1), size(W,2));
if ~isempty(Xval_lda)
    fprintf('Validation: %d×%d\n', size(Xval_lda,1), size(W,2));
else
    fprintf('Validation: not defined (2-way split)\n');
end
