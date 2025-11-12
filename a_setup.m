% a_setup.m — setup that always uses the project root
clearvars; close all; clc;
rng(1234,'twister');

st = dbstack('-completenames');
if ~isempty(st)
    thisFileDir = fileparts(st(1).file);
else
    try
        thisFileDir = fileparts(matlab.desktop.editor.getActiveFilename);
    catch
        thisFileDir = pwd;
    end
end

% find for /src
if endsWith(thisFileDir, [filesep 'src'])
    projectRoot = fileparts(thisFileDir);
elseif isfolder(fullfile(thisFileDir,'src'))
    projectRoot = thisFileDir;
else
    % walk up until a 'src' is found (max 5 levels)
    tmp = thisFileDir; found = false;
    for k=1:5
        tmp = fileparts(tmp);
        if isfolder(fullfile(tmp,'src')), projectRoot = tmp; found = true; break; end
    end
    if ~found, error('Could not locate project root (folder containing "src").'); end
end

cd(projectRoot);
addpath(genpath(fullfile(projectRoot,'src')));

req = {"splits","features/raw","features/hog","models/nn"};
for i = 1:numel(req)
    p = fullfile(projectRoot, req{i});
    if ~exist(p,'dir'), mkdir(p); end
end
disp(['Setup complete at: ' projectRoot]);