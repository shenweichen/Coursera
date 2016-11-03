% This file was published on Wed Nov 14 20:48:30 2012, UTC.

function a4_main(n_hid, lr_rbm, lr_classification, n_iterations)
% first, train the rbm
    global report_calls_to_sample_bernoulli
    report_calls_to_sample_bernoulli = false;
    global data_sets
    if prod(size(data_sets)) ~= 1,
        error('You must run a4_init before you do anything else.');
    end
    rbm_w = optimize([n_hid, 256], ...
                     @(rbm_w, data) cd1(rbm_w, data.inputs), ...  % discard labels
                     data_sets.training, ...
                     lr_rbm, ...
                     n_iterations);
    % rbm_w is now a weight matrix of <n_hid> by <number of visible units, i.e. 256>
    show_rbm(rbm_w);
    input_to_hid = rbm_w;
    % calculate the hidden layer representation of the labeled data
    hidden_representation = logistic(input_to_hid * data_sets.training.inputs);
    % train hid_to_class
    data_2.inputs = hidden_representation;
    data_2.targets = data_sets.training.targets;
    hid_to_class = optimize([10, n_hid], @(model, data) classification_phi_gradient(model, data), data_2, lr_classification, n_iterations);
    % report results
    for data_details = reshape({'training', data_sets.training, 'validation', data_sets.validation, 'test', data_sets.test}, [2, 3]),
        data_name = data_details{1};
        data = data_details{2};
        hid_input = input_to_hid * data.inputs; % size: <number of hidden units> by <number of data cases>
        hid_output = logistic(hid_input); % size: <number of hidden units> by <number of data cases>
        class_input = hid_to_class * hid_output; % size: <number of classes> by <number of data cases>
        class_normalizer = log_sum_exp_over_rows(class_input); % log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
        log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]); % log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
        error_rate = mean(double(argmax_over_rows(class_input) ~= argmax_over_rows(data.targets))); % scalar
        loss = -mean(sum(log_class_prob .* data.targets, 1)); % scalar. select the right log class probability using that sum; then take the mean over all data cases.
        fprintf('For the %s data, the classification cross-entropy loss is %f, and the classification error rate (i.e. the misclassification rate) is %f\n', data_name, loss, error_rate);
    end
    report_calls_to_sample_bernoulli = true;
end

function d_phi_by_d_input_to_class = classification_phi_gradient(input_to_class, data)
% This is about a very simple model: there's an input layer, and a softmax output layer. There are no hidden layers, and no biases.
% This returns the gradient of phi (a.k.a. negative the loss) for the <input_to_class> matrix.
% <input_to_class> is a matrix of size <number of classes> by <number of input units>.
% <data> has fields .inputs (matrix of size <number of input units> by <number of data cases>) and .targets (matrix of size <number of classes> by <number of data cases>).
% first: forward pass
    class_input = input_to_class * data.inputs; % input to the components of the softmax. size: <number of classes> by <number of data cases>
    class_normalizer = log_sum_exp_over_rows(class_input); % log(sum(exp)) is what we subtract to get normalized log class probabilities. size: <1> by <number of data cases>
    log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]); % log of probability of each class. size: <number of classes> by <number of data cases>
    class_prob = exp(log_class_prob); % probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes> by <number of data cases>
    % now: gradient computation
    d_loss_by_d_class_input = -(data.targets - class_prob) ./ size(data.inputs, 2); % size: <number of classes> by <number of data cases>
    d_loss_by_d_input_to_class = d_loss_by_d_class_input * data.inputs.'; % size: <number of classes> by <number of input units>
    d_phi_by_d_input_to_class = -d_loss_by_d_input_to_class;
end
 
function indices = argmax_over_rows(matrix)
    [dump, indices] = max(matrix);
end

function ret = log_sum_exp_over_rows(matrix)
  % This computes log(sum(exp(a), 1)) in a numerically stable way
  maxs_small = max(matrix, [], 1);
  maxs_big = repmat(maxs_small, [size(matrix, 1), 1]);
  ret = log(sum(exp(matrix - maxs_big), 1)) + maxs_small;
end

