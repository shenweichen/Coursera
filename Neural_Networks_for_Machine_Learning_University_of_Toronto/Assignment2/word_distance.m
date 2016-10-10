function distance = word_distance(word1, word2, model)
% Shows the L2 distance between word1 and word2 in the word_embedding_weights.
% Inputs:
%   word1: The first word as a string.
%   word2: The second word as a string.
%   model: Model returned by the training script.
% Example usage:
%   word_distance('school', 'university', model);

word_embedding_weights = model.word_embedding_weights;
vocab = model.vocab;
id1 = strmatch(word1, vocab, 'exact');
id2 = strmatch(word2, vocab, 'exact');
if ~any(id1)
  fprintf(1, 'Word ''%s\'' not in vocabulary.\n', word1);
  return;
end
if ~any(id2)
  fprintf(1, 'Word ''%s\'' not in vocabulary.\n', word2);
  return;
end
word_rep1 = word_embedding_weights(id1, :);
word_rep2 = word_embedding_weights(id2, :);
diff = word_rep1 - word_rep2;
distance = sqrt(sum(diff .* diff));
