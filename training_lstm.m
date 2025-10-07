% Define Absolute Paths
rpwFolder = 'E:\teja important\DataSet\rpw';  
nrpwFolder = 'E:\teja important\DataSet\nrpw';  

% Get Audio Files
rpwFiles = dir(fullfile(rpwFolder, '*.wav'));  
nrpwFiles = dir(fullfile(nrpwFolder, '*.wav'));

% Ensure Data Exists
if isempty(rpwFiles) || isempty(nrpwFiles)
    error('Error: Audio data missing. Ensure "rpw" and "nrpw" folders contain .wav files.');
end

% Load and Extract Features
XTrain = {};
YTrain = [];

for i = 1:length(rpwFiles)
    [audio, fs] = audioread(fullfile(rpwFiles(i).folder, rpwFiles(i).name));
    features = extractSpectFeatures(audio, fs);
    XTrain{end+1} = features; % Keep as [257 × 300]
    YTrain(end+1,1) = 1;  % RPW class
end

for i = 1:length(nrpwFiles)
    [audio, fs] = audioread(fullfile(nrpwFiles(i).folder, nrpwFiles(i).name));
    features = extractSpectFeatures(audio, fs);
    XTrain{end+1} = features; % Keep as [257 × 300]
    YTrain(end+1,1) = 0;  % Non-RPW class
end

% Convert Labels to Categorical
YTrain = categorical(YTrain);

% Balance Dataset Using Oversampling
numClass0 = sum(YTrain == '0');
numClass1 = sum(YTrain == '1');

if numClass0 > numClass1
    idx = find(YTrain == '1');
    extraSamples = datasample(idx, numClass0 - numClass1, 'Replace', true);
    XTrain = [XTrain, XTrain(extraSamples)];
    YTrain = [YTrain; YTrain(extraSamples)];
elseif numClass1 > numClass0
    idx = find(YTrain == '0');
    extraSamples = datasample(idx, numClass1 - numClass0, 'Replace', true);
    XTrain = [XTrain, XTrain(extraSamples)];
    YTrain = [YTrain; YTrain(extraSamples)];
end

% Convert YTrain to Column Vector (Fixing Dimension Issue)
YTrain = reshape(YTrain, [], 1);

% Train-Test Split (80% Train, 20% Test)
numSamples = length(YTrain);
cv = cvpartition(numSamples, 'HoldOut', 0.2);
XTrainData = XTrain(training(cv));  
YTrainData = YTrain(training(cv));  
XTestData = XTrain(test(cv));  
YTestData = YTrain(test(cv));  

% Define LSTM Model
layers = [
    sequenceInputLayer(257, 'Name', 'input')  % Ensure input size matches feature dimension
    lstmLayer(128, 'OutputMode', 'sequence', 'Name', 'lstm1')
    dropoutLayer(0.3, 'Name', 'dropout1')  % Prevent overfitting
    batchNormalizationLayer('Name', 'batchNorm1') % Normalize activations
    lstmLayer(64, 'OutputMode', 'last', 'Name', 'lstm2')
    dropoutLayer(0.3, 'Name', 'dropout2')
    batchNormalizationLayer('Name', 'batchNorm2')
    fullyConnectedLayer(2, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 25, ...  % Increase epochs
    'InitialLearnRate', 0.0001, ...  % Lower learning rate
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto'); 

% Train Model
net = trainNetwork(XTrainData, YTrainData, layers, options);

% Save Model
save('rpwclassify2.mat', 'net');

% Evaluate Model
YPred = classify(net, XTestData);
accuracy = sum(YPred == YTestData) / numel(YTestData);
disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);

% Confusion Matrix
figure;
cm = confusionchart(YTestData, YPred);
cm.Title = 'Confusion Matrix';

% Precision, Recall, F1-Score
tp = sum((YPred == '1') & (YTestData == '1'));
fp = sum((YPred == '1') & (YTestData == '0'));
fn = sum((YPred == '0') & (YTestData == '1'));

% Handle division by zero
precision = tp / (tp + fp + eps);
recall = tp / (tp + fn + eps);
f1 = 2 * (precision * recall) / (precision + recall + eps);

fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1-Score: %.2f\n', f1);
