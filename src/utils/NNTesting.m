function prediction = NNTesting(testFeature, modelNN)
trainX = double(modelNN.neighbours);
trainY = double(modelNN.labels(:));

x = double(testFeature(:))';
d2 = sum((trainX - x).^2,2);
[~,idx] = min(d2);
prediction = trainY(idx);
end
