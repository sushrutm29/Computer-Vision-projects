%%IMAGE_ASSOCIATION Function to form clusters of a given set of images
function image_association_1(radius, n_bins, conv_threshold)
    %% Setting default params
    if nargin == 0 %sets default radius length and bin count
        radius = 0.32;
        n_bins = 8;
        conv_threshold = 5;
    end
 
    if nargin == 1
        n_bins = 8;
        conv_threshold = 5;
    end
    
    if nargin == 2
        conv_threshold = 5;
    end
 
    %% Creating structure to store images
    images = struct([]);
 
    %% Reads training images
    fprintf("\nReading training images...");
    for i = 1 : 10
        imageName = "superset_1/entry_" + i + ".jpg";
        image = imread(imageName);
        images(i).image = image;
        images(i).name = imageName;
    end
    
    for i = 0 : 10
        imageName = "superset_1/fountain_" + i + ".jpg";
        image = imread(imageName);
        images(i+11).image = image;
        images(i+11).name = imageName;
    end
 
    for i = 0 : 7
        imageName = "superset_1/jesus_" + i + ".jpg";
        image = imread(imageName);
        images(i+22).image = image;
        images(i+22).name = imageName;
    end
    
    %% Creating structure to store histogram representations
    fprintf("\nCreating histograms (bins) from images...");
    reps_struct = create_histogram(images, n_bins);
    
    fprintf("\nPerforming mean_shift...");
    final_clusters = mean_shift(reps_struct, radius, n_bins, conv_threshold);
    
    fprintf("\n\nOutput:\n\n");
    for i=1:size(final_clusters, 2)
        current_cluster = final_clusters(i).images;
        
        for j=1:size(current_cluster, 2)
            disp("Cluster "+i+" contains image "+current_cluster(j).name);
        end
    end
end

%%CREATE_HISTOGRAMS Function to form RGB histograms for input image
function representations = create_histogram(images, n_bins)
    representations = struct([]);

    [n_rows, n_cols, ~] = size(images(1).image);
    
    const = double(n_rows * n_cols);
    
    bin_size = double(256 / n_bins);
    edges = zeros(1, n_bins);
    edges(1, 1) = 1;

    for i=1:n_bins
        edges(1, i+1) = bin_size*i;
    end
    
    %% Computing bin strength for each of the bins
    for i=1:size(images, 2)
        representations(i).name = images(i).name;
        representations(i).rep = struct([]);
        
        % For red channel
        X = images(i).image(:,:,1);
        bins_r = double(histcounts(X, edges));
        
        % For green channel
        X = images(i).image(:,:,2);
        bins_g = double(histcounts(X, edges));
        
        % For blue channel
        X = images(i).image(:,:,3);
        bins_b = double(histcounts(X, edges));
        
        rep_count = 0;
        
        for j=1:size(bins_r, 2)
            rep_count = rep_count + 1;
            representations(i).rep(rep_count).value = bins_r(j)/const;
        end
        
        for j=1:size(bins_g, 2)
            rep_count = rep_count + 1;
            representations(i).rep(rep_count).value = bins_g(j)/const;
        end
        
        for j=1:size(bins_b, 2)
            rep_count = rep_count + 1;
            representations(i).rep(rep_count).value = bins_b(j)/const;
        end
    end
end

%%MEAN_SHIFT Function to perform meanshift on a set of histograms
function image_groups = mean_shift(hist_struct, radius, n_bins, conv_threshold)
    %% Scan through each image and compute clusters
    %structure used to store different image groups
    image_groups = struct([]);
    prev_centroids = struct([]);
   
    for i = 1 : size(hist_struct, 2) %29 images total
        current_image = hist_struct(i).rep;  %the current image
        image_groups(i).images = struct([]);
        image_groups(i).centroid = struct([]);
        prev_centroids(i).bin = struct([]);

        % Assign intital value to centroid in N-Dimensions
        for j=1:n_bins*3
            image_groups(i).centroid(j).value = current_image(j).value;
            prev_centroids(i).bin(j).value = current_image(j).value;
        end

        image_index = double(0); %number of images in the current image group
        %compares the current image with other images

        for j = 1 : size(hist_struct, 2)
            other_image = hist_struct(j).rep;
            other_name = hist_struct(j).name;
            
            if (j == i) %skips the current image
                continue;
            end
            %calculates the N-Dimensional distance of current image to other images
            distance_diff = 0;
            
            % Distance for red channel
            for k = 1 : n_bins*3
                distance_diff = distance_diff...
                    + (current_image(k).value ...
                    - other_image(k).value)^2;
            end
            
            new_dist = sqrt(distance_diff);
            
            %compare the distance against the radius
            if new_dist <= radius
                image_index = image_index + double(1);
                %put this image into the image_groups structure
                image_groups(i).images(image_index).rep = other_image;
                image_groups(i).images(image_index).name = other_name;
            end        
        end
    end
    
    % Repeat until clusters converge
    converged = false;

    while converged == false     
        %calculates new centroid by calculating the mean of each dimension
        %in the N-Dimensions (RGB values/number of images within current cluster)
        for i=1:n_bins*3
            for j=1:size(image_groups, 2)
                current_group = image_groups(j).images;
                sum_bin_val = double(0);
                
                for k=1:size(current_group, 2)
                    current_image = current_group(k).rep;
                    sum_bin_val = sum_bin_val + double(current_image(i).value);
                end
                
                prev_centroids(j).bin(i).value = image_groups(j).centroid(i).value;
                image_groups(j).centroid(i).value = double(sum_bin_val / size(current_group, 2));
            end
        end
        
        % Assign images to new clusters
        for j=1:size(image_groups, 2)
            image_groups(j).images = struct([]);
            image_index = 0;
            
            for i=1:size(hist_struct, 2)
                current_image = hist_struct(i).rep;  %the current image
                current_name = hist_struct(i).name;  %the current image name
                distance_diff = 0;
                
                for k=1:n_bins*3
                    distance_diff = distance_diff ...
                        + (current_image(k).value ...
                        - image_groups(j).centroid(k).value)^2;
                end
                
                if sqrt(distance_diff) <= radius
                    image_index = image_index + 1;
                    image_groups(j).images(image_index).rep = current_image;
                    image_groups(j).images(image_index).name = current_name;
                end
            end
        end
        
        % Compute distance moved by centroids and check convergence
        converged = true;
        
        for i=1:size(image_groups, 2)
            
            % Compute distance moved by centroid
            movement = double(0);

            for j=1:n_bins*3
                movement = movement + ...
                    (image_groups(i).centroid(j).value - prev_centroids(i).bin(j).value)^2;
            end
            
            % Check if converged (repeat if any centroid hasn't converged)
            if movement > conv_threshold
                converged = false;
                break;
            end
        end
    end
    
    % Merge clusters that are close to each other (because we don't want 29 clusters at the end)
    fprintf("\nMerging nearby clusters...");
    image_groups = merge_clusters(image_groups, 0.22, n_bins);
    
    for i=1:size(image_groups, 2)
        image_groups(i).images = struct([]);
    end
    
    % Final iteration through all images to assign them to closest cluster
    % after merging
    fprintf("\nFinal image assignment to clusters...");
    for i=1:size(hist_struct, 2)
        min_cluster = 1;
        min_dist = inf;
        
        for j=1:size(image_groups, 2)
            curr_dist = double(0);
            
            for k=1:n_bins*3
                curr_dist = curr_dist + (image_groups(j).centroid(k).value - hist_struct(i).rep(k).value)^2;
            end
            
            curr_dist = sqrt(curr_dist);
            
            if curr_dist < min_dist
                min_dist = curr_dist;
                min_cluster = j;
            end
        end
        
        image_groups(min_cluster).images(size(image_groups(min_cluster).images, 2) + 1).name = hist_struct(i).name;
    end 
end

%%MERGED_CLUSTERS Function to merge clusters that are nearby
% This function merges clusters within a certain distance (merge_threshold)
% of each other
function merged_clusters = merge_clusters(image_groups, merge_threshold, n_bins)
    merged_clusters = struct([]);
    
    % Array to keep track of already merged clusters
    redundant = zeros(1, size(image_groups, 2));
    
    % Loop to compute euclidean distance and merge clusters within
    % threshold
    for i=1:size(image_groups, 2)
        if ismember(i, redundant)
            continue;
        end
        
        for j=1:size(image_groups, 2)
            if i==j || ismember(j, redundant)
                continue;
            end
            
            distance_diff = 0;
            
            for k=1:n_bins*3
                distance_diff = distance_diff ...
                    + (image_groups(i).centroid(k).value ...
                    - image_groups(j).centroid(k).value)^2;
            end
            
            if sqrt(distance_diff) <= merge_threshold
                image_groups(i).images = merge_2_clusters(image_groups(i).images, image_groups(j).images);
                redundant(size(redundant, 2) + 1) = j;
            end
        end
    end
    
    % Final collection of merged clusters with centroid values
    for i=1:size(image_groups, 2)
        if ismember(i, redundant)
            continue;
        end
        
        merged_clusters(size(merged_clusters, 2) + 1).images = image_groups(i).images;
        
        for j=1:n_bins*3
            merged_clusters(size(merged_clusters, 2)).centroid(j).value = image_groups(i).centroid(j).value;
        end
    end
end

%%MERGE_2_CLUSTERS Function to merge 2 clusters
% This function takes a set union of 2 clusters to output
% a cluster having all elements of both clusters without duplicates
function merged_cluster = merge_2_clusters(cluster_1, cluster_2)
    for i=1:size(cluster_2, 2)
        found = false;
        
        for j=1:size(cluster_1, 2)
            if isequal(cluster_1(j).name, cluster_2(i).name)
                found = true;
                break;
            end
        end
        
        if found == false
            cluster_1(size(cluster_1, 2) + 1).rep = cluster_2(i).rep;
            cluster_1(size(cluster_1, 2) + 1).name = cluster_2(i).name;
        end
    end
    
    merged_cluster = cluster_1;
end
