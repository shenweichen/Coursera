function display_nearest_words(word, model, k)
% Shows the k-nearest words to the query word.
% Inputs:
%   word: The query word as a string.
%   model: Model returned by the training script.
%   k: The number of nearest words to display.
% Example usage:
%   display_nearest_words('school', model, 10);

word_embedding_weights = model.word_embedding_weights;
vocab = model.vocab;
id = strmatch(word, vocab, 'exact');
if ~any(id)
  fprintf(1, 'Word ''%s\'' not in vocabulary.\n', word);
  return;
end
% Compute distance to every other word.
vocab_size = size(vocab, 2);
word_rep = word_embedding_weights(id, :);
diff = word_embedding_weights - repmat(word_rep, vocab_size, 1);
distance = sqrt(sum(diff .* diff, 2));

% Sort by distance.
[d, order] = sort(distance);
order = order(2:k+1);  % The nearest word is the query word itself, skip that.
for i = 1:k
  fprintf('%s %.2f\n', vocab{order(i)}, distance(order(i)));
end
