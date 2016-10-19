function [charAcc, wordAcc] = ScoreModel (words, imageModel, pairwiseModel, tripletList)
% This function runs the Markov network model end-to-end and computes the
% per-character and per-word accuracy on provided data.
%
% Input:
%   words: A cell array where words{i} is the struct array for the ith
%     word (this is the structure of the provided 'allWords' data).
%   imageModel: The provided image model struct.
%   pairwiseModel: A K-by-K matrix (K is the alphabet size) where pairwiseModel(i,j)
%     is the factor value for the pairwise factor of character i followed by
%     character j.
%   tripletModel: The array of character triplets we will consider (along
%     with their corresponding factor values).
%
% Output:
%   charAcc: The percentage of all characters (across all words) correctly
%     identified. (Between 0 and 1)
%   wordAcc: The percentage of the words in which every character is
%     correctly identified. (Between 0 and 1)
%
% Copyright (C) Daphne Koller, Stanford University, 2012

predictions = ComputeWordPredictions(words, imageModel, pairwiseModel, tripletList);
[charAcc, wordAcc] = ScorePredictions(words, predictions, true);

end
