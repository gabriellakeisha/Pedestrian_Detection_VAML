% d5_features_hog_lbp.m — HOG+LBP concatenation
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

posDir = 'data/images/pos';
negDir = 'data/images/neg';
targetSize = [128 64];

hog_cell = [8 8]; hog_block = [2 2]; hog_bins = 9;
lbp_cell = [16 16];

fprintf('=== Extracting FUSION (HOG+LBP) features ===\n');
[X_fuse, y, fileNames] = loadPedestrianDataset(posDir, negDir, 'fusion_hog_lbp', ...
                                               targetSize, hog_cell, hog_block, hog_bins);

if ~exist('features/fusion','dir'), mkdir('features/fusion'); end
save('features/fusion/features_fusion_hog_lbp.mat', ...
     'X_fuse','y','fileNames','targetSize','hog_cell','hog_block','hog_bins','lbp_cell','-v7.3');
fprintf('FUSION features saved: %d × %d\n', size(X_fuse,1), size(X_fuse,2));
