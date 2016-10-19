function [charAcc, wordAcc] = ScorePredictions (words, predictions, showOutput)
% This function computes the character and word accuracies for a list of
% words and the predictions for each of those words.
%
% Input:
%   words: A cell array of struct arrays such as the provided 'allWords'.
%   predictions: A cell array of arrays. predictions{i} should be an array
%     of the character values of word i. This will be the return value of
%     ComputeWordPredictions.m when correctly implemented.
%   showOutput [optional]: Boolean, default true. If true, this functional
%     will print diagnostic information to the console as it runs.
%     Otherwise, there is no printed output.
%
% Output:
%   charAcc: The percentage of all characters (across all words) correctly
%     identified. (Between 0 and 1)
%   wordAcc: The percentage of the words in which every character is
%     correctly identified. (Between 0 and 1)
%
% Copyright (C) Daphne Koller, Stanford University, 2012


if (nargin < 3)
    showOutput = true;
end

numWords = length(words);
if (numWords ~= length(predictions))
    error('words and predictions must be same length');
end

totalChars = 0;
totalWords = 0;
totalCharsRight = 0;
totalWordsRight = 0;

for i = 1:numWords
    totalChars = totalChars + length(predictions{i});
    totalWords = totalWords + 1;
    charsRight = sum(predictions{i}(:) == vertcat(words{i}(:).groundTruth));
    totalCharsRight = totalCharsRight + charsRight;
    if (charsRight == length(predictions{i}))
        totalWordsRight = totalWordsRight + 1;
    end
    
    if (showOutput)
        charsWrong = length(predictions{i}) - charsRight;
        rightWord = char(horzcat(words{i}(:).groundTruth) + 'a' - 1);
        predWord = char(horzcat(predictions{i}(:)) + 'a' - 1);
    
        fprintf('%d\n  Correct:   %s\n  Predicted: %s\n  (%d mistaken characters)\n', ...
           i, rightWord, predWord, charsWrong);
    end
end

charAcc = totalCharsRight / totalChars;
wordAcc = totalWordsRight / totalWords;

if (showOutput)
    fprintf('\n\n %d / %d characters (%.2f%% accuracy)\n %d / %d words (%.2f%% accuracy)\n\n', ...
        totalCharsRight, totalChars, charAcc * 100, totalWordsRight, totalWords, wordAcc * 100);
end

end

