% b_halfhalf_splits.m — create 50/50 training/testing splits
clearvars; close all; clc; rng(1234,'twister');

st = dbstack('-completenames');
thisFileDir = fileparts(st(1).file);
projectRoot = fileparts(thisFileDir);
cd(projectRoot);

posDir = fullfile(projectRoot,'images','pos');
negDir = fullfile(projectRoot,'images','neg');

P = [dir(fullfile(posDir,'*.png')); dir(fullfile(posDir,'*.jpg')); dir(fullfile(posDir,'*.jpeg')); ...
     dir(fullfile(posDir,'*.PNG')); dir(fullfile(posDir,'*.JPG')); dir(fullfile(posDir,'*.JPEG'))];
N = [dir(fullfile(negDir,'*.png')); dir(fullfile(negDir,'*.jpg')); dir(fullfile(negDir,'*.jpeg')); ...
     dir(fullfile(negDir,'*.PNG')); dir(fullfile(negDir,'*.JPG')); dir(fullfile(negDir,'*.JPEG'))];

Npos = numel(P); 
Nneg = numel(N);

assert(Npos>0 && Nneg>0, sprintf('Found Npos=%d, Nneg=%d. Check data/images/pos & neg.',Npos,Nneg));

total = Npos + Nneg;
order = randperm(total);

% 50/50 SPLIT
Ntrain = round(0.5*total);  % ← 50% for training
trainIdx = order(1:Ntrain);
testIdx  = order(Ntrain+1:end);  % ← 50% for testing

% Create splits directory if needed
if ~exist(fullfile(projectRoot,'splits'),'dir')
    mkdir(fullfile(projectRoot,'splits'));
end

save(fullfile(projectRoot,'splits','splits_50_50.mat'),'trainIdx','testIdx','Npos','Nneg');

fprintf('\n50/50 SPLIT CREATED\n');
fprintf('Total samples: %d (Npos=%d, Nneg=%d)\n', total, Npos, Nneg);
fprintf('Training: %d samples (50%%)\n', numel(trainIdx));
fprintf('Testing:  %d samples (50%%)\n', numel(testIdx));
