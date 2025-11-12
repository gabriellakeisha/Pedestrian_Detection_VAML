%% data_preprocessing.m - Image Preprocessing for Pedestrian Dataset
clearvars; close all; clc; rng(1234,'twister');
addpath(genpath('src'));

fprintf('DATA PREPROCESSING START\n');

inputDir = 'images/';
outputDir = 'images_preprocessed/';
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

imgFiles = dir(fullfile(inputDir, '**/*.jpg'));
fprintf('Found %d images\n', numel(imgFiles));

targetSize = [128 64];  % standard INRIA pedestrian window size

for i = 1:numel(imgFiles)
    inPath = fullfile(imgFiles(i).folder, imgFiles(i).name);
    outPath = fullfile(outputDir, imgFiles(i).name);
    
    % Load and convert to grayscale
    I = imread(inPath);
    if size(I,3) == 3
        I = rgb2gray(I);
    end
    
    % Resize to consistent window
    I = imresize(I, targetSize);
    
    % Histogram equalization (contrast normalisation)
    I = adapthisteq(I);
    
    % Gaussian smoothing (noise reduction)
    I = imgaussfilt(I, 0.5);
    
    imwrite(I, outPath);
end

fprintf('Preprocessing complete! Saved to %s\n', outputDir);
