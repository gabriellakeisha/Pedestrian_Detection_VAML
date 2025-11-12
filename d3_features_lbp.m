% d3_features_lbp.m — Extract LBP features
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

posDir = 'data/images/pos';
negDir = 'data/images/neg';
targetSize = [128 64];
cellSize = [16 16];   % coarser cells for LBP work well

fprintf('Extracting LBP features\n');
[X_lbp, y, fileNames] = loadPedestrianDataset(posDir, negDir, 'lbp', targetSize, cellSize);

if ~exist('features/lbp','dir'), mkdir('features/lbp'); end
save('features/lbp/features_lbp.mat','X_lbp','y','fileNames','targetSize','cellSize','-v7.3');
fprintf('LBP features saved: %d samples × %d dims\n', size(X_lbp,1), size(X_lbp,2));