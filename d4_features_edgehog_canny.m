% d4_features_edgehog.m — Canny + HOG features
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

posDir = 'data/images/pos';
negDir = 'data/images/neg';
targetSize = [128 64];
cellSize = [8 8];
blockSize = [2 2];
numBins = 9;

fprintf('=== Extracting EDGEHOG (Canny->HOG) features ===\n');
[X_edgehog, y, fileNames] = loadPedestrianDataset( ...
    posDir, negDir, 'edgehog', targetSize, cellSize, blockSize, numBins);

if isempty(X_edgehog)
    error('No features extracted. Check folder paths and function support for ''edgehog''.');
end

if ~exist('features/edge','dir'), mkdir('features/edge'); end
save('features/edge/features_edgehog.mat', ...
     'X_edgehog','y','fileNames','targetSize','cellSize','blockSize','numBins','-v7.3');

fprintf('EDGEHOG features saved: %d samples × %d dims\n', size(X_edgehog,1), size(X_edgehog,2));