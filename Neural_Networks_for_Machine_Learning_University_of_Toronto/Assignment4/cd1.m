function ret = cd1(rbm_w, visible_data)
    % q9
    visible_data = sample_bernoulli(visible_data);
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    hidden_prob = visible_state_to_hidden_probabilities(rbm_w,visible_data);
    sample_hidden_data = sample_bernoulli(hidden_prob);
    grident1 = configuration_goodness_gradient(visible_data,sample_hidden_data);
   
    %reconstruction
    visible_prob = hidden_state_to_visible_probabilities(rbm_w,sample_hidden_data);
    sample_visible_data = sample_bernoulli(visible_prob);
    
    hidden_prob = visible_state_to_hidden_probabilities(rbm_w,sample_visible_data);
    % imporve--q8
    %sample_hidden_data=sample_bernoulli(hidden_prob);
    grident2 = configuration_goodness_gradient(sample_visible_data,hidden_prob);
    
    ret = grident1 - grident2;
 
    
end
