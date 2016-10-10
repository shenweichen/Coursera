function [embedding_layer_state, hidden_layer_state, output_layer_state] = ...
  fprop(input_batch, word_embedding_weights, embed_to_hid_weights,...
  hid_to_output_weights, hid_bias, output_bias)
% This method forward propagates through a neural network.
% Inputs:
%   input_batch: The input data as a matrix of size numwords X batchsize where,
%     numwords is the number of words, batchsize is the number of data points.
%     So, if input_batch(i, j) = k then the ith word in data point j is word
%     index k of the vocabulary.
%
%   word_embedding_weights: Word embedding as a matrix of size
%     vocab_size X numhid1, where vocab_size is the size of the vocabulary
%     numhid1 is the dimensionality of the embedding space.
%
%   embed_to_hid_weights: Weights between the word embedding layer and hidden
%     layer as a matrix of soze numhid1*numwords X numhid2, numhid2 is the
%     number of hidden units.
%
%   hid_to_output_weights: Weights between the hidden layer and output softmax
%               unit as a matrix of size numhid2 X vocab_size
%
%   hid_bias: Bias of the hidden layer as a matrix of size numhid2 X 1.
%
%   output_bias: Bias of the output layer as a matrix of size vocab_size X 1.
%
% Outputs:
%   embedding_layer_state: State of units in the embedding layer as a matrix of
%     size numhid1*numwords X batchsize
%
%   hidden_layer_state: State of units in the hidden layer as a matrix of size
%     numhid2 X batchsize
%
%   output_layer_state: State of units in the output layer as a matrix of size
%     vocab_size X batchsize
%

[numwords, batchsize] = size(input_batch);
[vocab_size, numhid1] = size(word_embedding_weights);
numhid2 = size(embed_to_hid_weights, 2);

%% COMPUTE STATE OF WORD EMBEDDING LAYER.
% Look up the inputs word indices in the word_embedding_weights matrix.
embedding_layer_state = reshape(...
  word_embedding_weights(reshape(input_batch, 1, []),:)',...
  numhid1 * numwords, []);

%% COMPUTE STATE OF HIDDEN LAYER.
% Compute inputs to hidden units.
inputs_to_hidden_units = embed_to_hid_weights' * embedding_layer_state + ...
  repmat(hid_bias, 1, batchsize);

% Apply logistic activation function.
% FILL IN CODE. Replace the line below by one of the options.
hidden_layer_state = 1 ./ (1 + exp(-inputs_to_hidden_units));
% Options
% (a) hidden_layer_state = 1 ./ (1 + exp(inputs_to_hidden_units));
% (b) hidden_layer_state = 1 ./ (1 - exp(-inputs_to_hidden_units));
% (c) hidden_layer_state = 1 ./ (1 + exp(-inputs_to_hidden_units));
% (d) hidden_layer_state = -1 ./ (1 + exp(-inputs_to_hidden_units));

%% COMPUTE STATE OF OUTPUT LAYER.
% Compute inputs to softmax.
% FILL IN CODE. Replace the line below by one of the options.
 inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, 1, batchsize);
% Options
% (a) inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, 1, batchsize);
% (b) inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, batchsize, 1);
% (c) inputs_to_softmax = hidden_layer_state * hid_to_output_weights' +  repmat(output_bias, 1, batchsize);
% (d) inputs_to_softmax = hid_to_output_weights * hidden_layer_state +  repmat(output_bias, batchsize, 1);

% Subtract maximum. 
% Remember that adding or subtracting the same constant from each input to a
% softmax unit does not affect the outputs. Here we are subtracting maximum to
% make all inputs <= 0. This prevents overflows when computing their
% exponents.
inputs_to_softmax = inputs_to_softmax...
  - repmat(max(inputs_to_softmax), vocab_size, 1);

% Compute exp.
output_layer_state = exp(inputs_to_softmax);

% Normalize to get probability distribution.
output_layer_state = output_layer_state ./ repmat(...
  sum(output_layer_state, 1), vocab_size, 1);
