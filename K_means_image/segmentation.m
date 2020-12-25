%Performs pixel-wise segmentation onto the training images with tractor 
%cut out. Reads in three castle_p30 training images and generate a
%classifier using k-means algorithm with k-value set to 10. 
function segmentation()
    %%reads in training images from castle-P30
    trainImageNames = ["castle_p30_1", "castle_p30_14", "castle_p30_23"];
    %%initializes tractor/non-tractor clusters arra
    tractorClusters = [];   
    nonTractorClusters = [];
    k_value = 10;
    avgMax_y = -1;  %average maximum y value of the tractor
    avgMin_y = Inf; %average minimum y value of the tractor
    %%creates the tractor/non-tractor clusters with training images.
    for i = 1 : size(trainImageNames, 2)
        fprintf("\n====================");
        currentName = "castle-P30/" + trainImageNames(i);
        trainingImgName = "castle-P30/" + trainImageNames(i);
        fprintf("\ncurrentName = %s", currentName);
        %creates tractor/non-tractor clusters for the current image.
        [max_y, min_y, tractor_matrix, non_tractor_matrix] = ...
            createSets(currentName + ".jpg", trainingImgName + ".png");
        avgMax_y = max(avgMax_y, max_y);
        avgMin_y = min(avgMin_y, min_y);
        %applies k-means to the two clusters
        [~, tractorImg] = kmeans(tractor_matrix, k_value,...
        'MaxIter', 500, "EmptyAction", "singleton");
        fprintf("\nfinish tractorImg kmeans!!!");
        [~, nonTractorImg] = kmeans(non_tractor_matrix, k_value,...
        'MaxIter', 500, "EmptyAction", "singleton");
        fprintf("\nfinish nonTractorImg kmeans!!!");
        %concatenates the clusters to the cluster arrays
        tractorClusters = cat(1, tractorClusters, tractorImg);
        nonTractorClusters = cat(1, nonTractorClusters, nonTractorImg);
    end
    %increase the average minimum y range by 50 pixels
    avgMin_y = avgMin_y - 50;
    
    %%process the testing images from castle-P19 dataset
    test_img_count = 18;    %total number of testing images
    testImgName = "castle-P19/castle_";
    for i = 0 : test_img_count
        currentFileName = testImgName + i;
        fprintf("\ncurrent testing image = %s", currentFileName);
        currentImg = double(imread(currentFileName + ".jpg"));
        [row_size, col_size, ~] = size(currentImg);
        %%calculates the shortest distance from current pixel to both
        %%tractor and non-tractor centroids
        for j = 1 : row_size
           for k = 1 : col_size
               c_red = currentImg(j, k, 1);
               c_green = currentImg(j, k, 2);
               c_blue = currentImg(j, k, 3);
               
               tractorDist = calculateShortestDist(tractorClusters, c_red, ...
                    c_green, c_blue);
               nonTractorDist = calculateShortestDist(nonTractorClusters, ...
                    c_red, c_green, c_blue);
               
               %determines if current pixel is tractor pixel or not, if 
               %it is then paints it blue
               if ~(tractorDist <= nonTractorDist && avgMax_y > j && ...
                       avgMin_y < j)
                   currentImg(j, k, 1) = 0;
                   currentImg(j, k, 2) = 0;
                   currentImg(j, k, 3) = 255;
               end
           end
        end
        
        %save each current test image
        newFileName = currentFileName + "_mask.jpg";
        %Converts the output image to 8-bit unsigned integer arrays
        currentImg = uint8(currentImg);
        imwrite(currentImg, newFileName);
    end
end

%calculates the shortest distance between two points using their RGB
%values.
%Inputs:
%   c_matrix: matrix contains centroid points' RGB values.
%   c_red: red value of the original pixel. 
%   c_green: green value of the original pixel. 
%   c_blue: blue value of the original pixel. 
function shortestDist = calculateShortestDist(c_matrix, c_red, c_green,...
            c_blue)
    shortestDist = Inf; %shortest distance from current pixel to a cluster
    for x = 1 : size(c_matrix, 1)
        t_red = c_matrix(x, 1);
        t_green = c_matrix(x, 2);
        t_blue = c_matrix(x, 3);
        
        %calculates distance using Euclidean formula
        rgbDist = sqrt((c_red - t_red) ^ 2 + (c_green - t_green) ^ 2 ...
            + (c_blue - t_blue) ^ 2);
        %found shorter distance
        if shortestDist > rgbDist
            shortestDist = rgbDist;
        end
    end
end

%Creates the tractor and non-tractor pixel sets using the original 
%training image.
%Inputs:
%   trainImg: file name of the original training image without mask color.
%   maskTrainImg: file name of the original training image with mask color.
%Outputs:
%   max_y: maximum y coordinate value of the tractor section
%   min_y: minimum y coordinate value of the tractor section
%   tractor_matrix: matrix contains tractor centroid points' RGB values.
%   non_tractor_matrix: matrix contains non-tractor centroid points' 
%                         RGB values.
function [max_y, min_y, tractor_matrix, non_tractor_matrix] = ...
                createSets(trainImgName, maskImgName)
    %reads in the images
    trainImg = imread(trainImgName);
    trainImg = double(trainImg);
    maskTrainImg = imread(maskImgName);
    maskTrainImg = double(maskTrainImg);
    [row_size, col_size, ~] = size(maskTrainImg);
    tractor_count = 1;  %number of tractor pixels
    non_tractor_count = 1;  %number of non-tractor pixels
    %initializes and pre-allocate spaces for the tractor/non-tractor
    %matrices
    maxSize = row_size * col_size;
    tractor_matrix = zeros(maxSize, 3);
    non_tractor_matrix = zeros(maxSize, 3);
    max_y = -1;
    min_y = Inf;
    %scans through the training image and labels each pixel as
    %tractor/non-tractor
    for i = 1 : row_size
       for j = 1 : col_size
           %checks if current pixel is tractor or not
           if maskTrainImg(i, j, 1) == 255 &&...    %tractor pixel
                   maskTrainImg(i, j, 2) == 0 &&...
                   maskTrainImg(i, j, 3) == 0
               tractor_matrix(tractor_count, :) = ...
                   [trainImg(i, j, 1) trainImg(i, j, 2)...
                   trainImg(i, j, 3)];
               tractor_count = tractor_count + 1;
               max_y = max(max_y, i);
               min_y = min(min_y, i);
           else     %non-tractor pixel
               non_tractor_matrix(non_tractor_count, :) = ...
                       [trainImg(i, j, 1) trainImg(i, j, 2)...
                       trainImg(i, j, 3)];
               non_tractor_count = non_tractor_count + 1;
           end
       end
    end
    %remove extra pre-allocated space in matrices
    tractor_matrix = tractor_matrix(1 : tractor_count, :);
    non_tractor_matrix = non_tractor_matrix(1 : non_tractor_count, :);
end