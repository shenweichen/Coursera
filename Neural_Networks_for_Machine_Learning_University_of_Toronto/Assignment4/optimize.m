function model = optimize(model_shape, gradient_function, training_data, learning_rate, n_iterations)
% This trains a model that's defined by a single matrix of weights.
% <model_shape> is the shape of the array of weights.
% <gradient_function> is a function that takes parameters <model> and <data> and returns the gradient (or approximate gradient in the case of CD-1) of the function that we're maximizing. Note the contrast with the loss function that we saw in PA3, which we were minimizing. The returned gradient is an array of the same shape as the provided <model> parameter.
% This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.
% This returns the matrix of weights of the trained model.
    model = (a4_rand(model_shape, prod(model_shape)) * 2 - 1) * 0.1;
    momentum_speed = zeros(model_shape);
    mini_batch_size = 100;
    start_of_next_mini_batch = 1;
    for iteration_number = 1:n_iterations,
        mini_batch = extract_mini_batch(training_data, start_of_next_mini_batch, mini_batch_size);
        start_of_next_mini_batch = mod(start_of_next_mini_batch + mini_batch_size, size(training_data.inputs, 2));
        gradient = gradient_function(model, mini_batch);
        momentum_speed = 0.9 * momentum_speed + gradient;
        model = model + momentum_speed * learning_rate;
    end
end

