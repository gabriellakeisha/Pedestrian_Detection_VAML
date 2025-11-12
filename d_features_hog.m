% d_features_hog.m — extract HOG features
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

posDir = 'images/pos';
negDir = 'images/neg';
targetSize = [128 64];
cellSize = [8 8];
blockSize = [2 2];
numBins = 9;

[X_hog, y, fileNames] = loadPedestrianDataset(posDir, negDir, 'hog', targetSize, cellSize, blockSize, numBins);
save('features/hog/features_hog.mat','X_hog','y','fileNames','targetSize','cellSize','blockSize','numBins','-v7.3');

fprintf('HOG features saved: %d samples × %d dimensions\n',size(X_hog,1),size(X_hog,2));
