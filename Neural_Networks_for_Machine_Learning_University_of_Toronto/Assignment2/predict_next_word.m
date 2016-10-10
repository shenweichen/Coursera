function predict_next_word(word1, word2, word3, model, k)
% Predicts the next word.
% Inputs:
%   word1: The first word as a string.
%   word2: The second word as a string.
%   word3: The third word as a string.
%   model: Model returned by the training script.
%   k: The k most probable predictions are shown.
% Example usage:
%   predict_next_word('john', 'might', 'be', model, 3);
%   predict_next_word('life', 'in', 'new', model, 3);

word_embedding_weights = model.word_embedding_weights;
vocab = model.vocab;
id1 = strmatch(word1, vocab, 'exact');
id2 = strmatch(word2, vocab, 'exact');
id3 = strmatch(word3, vocab, 'exact');
if ~any(id1)
  fprintf(1, 'Word ''%s\'' not in vocabulary.\n', word1);
  return;
end
if ~any(id2)
  fprintf(1, 'Word ''%s\'' not in vocabulary.\n', word2);
  return;
end
if ~any(id3)
  fprintf(1, 'Word ''%s\'' not in vocabulary.\n', word3);
  return;
end
input = [id1; id2; id3];
[embedding_layer_state, hidden_layer_state, output_layer_state] = ...
  fprop(input, model.word_embedding_weights, model.embed_to_hid_weights,...
        model.hid_to_output_weights, model.hid_bias, model.output_bias);
[prob, indices] = sort(output_layer_state, 'descend');
for i = 1:k
  fprintf(1, '%s %s %s %s Prob: %.5f\n', word1, word2, word3, vocab{indices(i)}, prob(i));
end
