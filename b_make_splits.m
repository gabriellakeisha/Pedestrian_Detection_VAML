% b_make_splits.m — create training/testing splits (saves to /splits)
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

Npos = numel(P); Nneg = numel(N);
assert(Npos>0 && Nneg>0, sprintf('Found Npos=%d, Nneg=%d. Check data/images/pos & neg.',Npos,Nneg));

total = Npos + Nneg;
order = randperm(total);
Ntrain = round(0.7*total); % 70
trainIdx = order(1:Ntrain);
testIdx  = order(Ntrain+1:end);

save(fullfile(projectRoot,'splits','splits.mat'),'trainIdx','testIdx','Npos','Nneg');
fprintf('Split created: %d train / %d test (Npos=%d, Nneg=%d)\n', numel(trainIdx), numel(testIdx), Npos, Nneg);
