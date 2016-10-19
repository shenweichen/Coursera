function allFactors = BuildOCRNetwork (images, imageModel, pairwiseModel, tripletList)
% This function constructs the Markov network for one word by putting
% together the full list of different types of factors.
%
% Input:
%   images: A struct array of the images for each character in the word.
%   imageModel: The provided image model.
%   pairwiseModel: The provided pairwise factor model.
%   tripletList: The provided triplet factor list.
%
% Output:
%   allFactors: An array of factor structs that constitutes the full Markov
%     network.
%
% NOTE:
% The default implementation of each ConstructFactors function returns an
% empty list, so as you implement each of those in succession, more and
% more factors will be added to the resulting network.
%
% You can control which factors are added after you've implemented all of
% the factor functions in the following way:
%
% Pairwise factors: pass in an empty array (ie, []) for the pairwiseModel,
% and no pairwise factors will be added.
%
% Triplet factors: pass in an empty array for the tripletList, and no
% triplet factors will be added.
%
% Similarity factors: if you set imageModel.ignoreSimilarity = 1, then the
% similarity factors will not be added (nb: this is a bit of a kludge.
% Apologies.)
%
% Copyright (C) Daphne Koller, Stanford University, 2012

if (~exist('pairwiseModel', 'var'))
    pairwiseModel = [];
end
if (~exist('tripletList', 'var'))
    tripletList = [];
end

singletonFactors = ComputeSingletonFactors(images, imageModel);

if (~isempty(pairwiseModel))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FOR STUDENTS:
    % Once you've tested out ComputeEqualPairwiseFactors, as discussed in
    % the assignment handout, you should comment out that line and
    % uncomment the following line to use the "real" pairwise factor
    % implementation (which you must provide.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %pairwiseFactors = ComputeEqualPairwiseFactors(images, imageModel.K);
    pairwiseFactors = ComputePairwiseFactors(images, pairwiseModel, imageModel.K);
else
    pairwiseFactors = [];
end

if (~isempty(tripletList))  
    tripletFactors = ComputeTripletFactors(images, tripletList, imageModel.K);
else
    tripletFactors = [];
end

% Your code here:

if (isfield(imageModel, 'ignoreSimilarity') && imageModel.ignoreSimilarity)
    simFactors = [];
else
    allSimFactors = ComputeAllSimilarityFactors(images, imageModel.K);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FOR STUDENTS:
    % In the provided implementation of ChooseTopSimilarityFactors, all
    % factors are returned, so no selection occurs. Once you try to run
    % inference using all factors (as described in the assignment handout),
    % you should implement the function so that the network contains only
    % the selected factors.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    simFactors = ChooseTopSimilarityFactors(allSimFactors, 2);
end

% This mess is necessary since Octave crashes if you try to vertcat onto a
% scalar, and it is possible that tripletFactors is scalar (for length 3
% words).
cellFactors{1} = singletonFactors;
cellFactors{2} = pairwiseFactors;
cellFactors{3} = tripletFactors;
cellFactors{4} = simFactors;
factorsPresent = ~[isempty(singletonFactors) isempty(pairwiseFactors) isempty(tripletFactors) isempty(simFactors)];

allFactors = vertcat(cellFactors{factorsPresent});

end

