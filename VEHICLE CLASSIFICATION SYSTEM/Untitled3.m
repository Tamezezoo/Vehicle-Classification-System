%% ========================================================================
%%                      VEHICLE CLASSIFICATION SYSTEM
%% ========================================================================
clc; clear; close all;

% ==========================================
% 1. System Configurations & Constants
% ==========================================
IMG_SIZE = [128, 128];
datasetPath = 'dataset';
categories = {'Cars', 'Trucks'};

fprintf('Categories detected: %s, %s\n', categories{1}, categories{2});

%% ------------------------------------------
%% Step 1 & 2: Data Loading & Split Management
%% ------------------------------------------
fprintf('\nStep 1 & 2: Loading Dataset & Splitting...\n');

% Leverage imageDatastore for automated image management and label assignment
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Filter dataset strictly to specified classes and enforce strict categorical ordering
imds = subset(imds, ismember(imds.Labels, categories));
imds.Labels = categorical(imds.Labels, categories); 

% Split data randomly (80% training, 20% testing) while maintaining class balance
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

fprintf('Total training images: %d\n', numel(imdsTrain.Files));
fprintf('Total testing images: %d\n', numel(imdsTest.Files));

%% ------------------------------------------
%% Step 3 & 4: Feature Extraction (HOG)
%% ------------------------------------------
fprintf('\nStep 3 & 4: Extracting Features for Training...\n');
[X_train, y_train] = processImagePipeline(imdsTrain, IMG_SIZE);

fprintf('Extracting Features for Testing...\n');
[X_test, y_test] = processImagePipeline(imdsTest, IMG_SIZE);

%% ------------------------------------------
%% Step 2: SVM Model Training
%% ------------------------------------------
fprintf('\nStep 5: Training SVM Classifier...\n');
svmModel = fitcsvm(X_train, y_train, ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint', 1.0, ...
    'KernelScale', 'auto');

fprintf('Model trained successfully.\n');

%% ------------------------------------------
%% Step 3: Model Evaluation
%% ------------------------------------------
fprintf('\nStep 6: Evaluating Model Performance\n');
y_pred = predict(svmModel, X_test);

% Calculate overall validation accuracy
accuracy = sum(y_pred == y_test) / numel(y_test) * 100;
fprintf('Overall Accuracy: %.2f%%\n', accuracy);

% Generate and display confusion matrix chart
figure('Name', 'Evaluation Metrics');
confusionchart(y_test, y_pred);
title('Vehicle Classification Confusion Matrix');

%% ------------------------------------------
%% Step 4: Inference on New Test Image
%% ------------------------------------------
fprintf('\nStep 7: Testing Model on External Image: test.jpg\n');
testImgPath = 'test.jpg';

if exist(testImgPath, 'file')
    % Read and process the target test image
    img = imread(testImgPath);
    imgResized = imresize(img, IMG_SIZE);
    
    if size(imgResized, 3) == 3
        gray = rgb2gray(imgResized);
    else
        gray = imgResized;
    end
    
    blurred = imgaussfilt(gray, 1.1); 
    segmented = edge(blurred, 'Canny'); 
    
    % Extract HOG features directly from the Canny edge map
    testFeatures = extractHOGFeatures(segmented, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9);
    
    % Perform prediction and parse the categorical output to string
    predLabel = predict(svmModel, testFeatures);
    predStr = char(predLabel);
    fprintf('Final Prediction Result: %s\n', predStr);
    
    % Resize original image and edge map to a uniform size for side-by-side display
    dispImg = imresize(img, [400, 400]);
    segImg = imresize(segmented, [400, 400]);
    
    % Convert binary edge map to RGB format to allow horizontal stitching
    segRGB = uint8(cat(3, segImg, segImg, segImg) * 255);
    
    % Set text color dynamically: Red for Trucks, Green for Cars
    if strcmp(predStr, 'Trucks')
        textColor = [255, 0, 0]; % Red
    else
        textColor = [0, 255, 0]; % Green
    end
    
    % Overlay the prediction label onto the main image
    dispImg = insertText(dispImg, [20, 50], ['PRED: ' predStr], ...
        'FontSize', 24, 'TextColor', textColor, 'BoxOpacity', 0);
    
    % Horizontally stack the original image and edge map for final visualization
    combinedView = [dispImg, segRGB];
    figure('Name', 'Final Project - Visualization');
    imshow(combinedView);
    title('Original Image (Predicted) vs Segmented Image (Canny)');
else
    fprintf('Error: Target file ''test.jpg'' could not be found in the specified path!\n');
end

%% ========================================================================
%% Helper Function: Image Preprocessing & Feature Extraction Pipeline
%% ========================================================================
function [DomainMask, labelsArray] = processImagePipeline(imds, imgSize)
    numImages = numel(imds.Files);
    labelsArray = imds.Labels;
    
    % Process the first image to dynamically determine HOG dimensions for pre-allocation
    tempImg = readimage(imds, 1);
    tempResized = imresize(tempImg, imgSize);
    if size(tempResized, 3) == 3, tempGray = rgb2gray(tempResized); else, tempGray = tempResized; end
    tempBlur = imgaussfilt(tempGray, 1.1);
    tempSeg = edge(tempBlur, 'Canny');
    sampleHOG = extractHOGFeatures(tempSeg, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9);
    
    numFeatures = numel(sampleHOG);
    DomainMask = zeros(numImages, numFeatures);
    
    % Iterate through the datastore to extract HOG features for all images
    for i = 1:numImages
        img = readimage(imds, i);
        imgResized = imresize(img, imgSize);
        
        if size(imgResized, 3) == 3
            gray = rgb2gray(imgResized);
        else
            gray = imgResized;
        end
        
        blurred = imgaussfilt(gray, 1.1);
        segmented = edge(blurred, 'Canny');
        
        DomainMask(i, :) = extractHOGFeatures(segmented, 'CellSize', [8 8], 'BlockSize', [2 2], 'NumBins', 9);
    end
end