global randomness_source
load a4_randomness_source

global data_sets
temp = load('data_set'); % same as in PA3
data_sets = temp.data;

global report_calls_to_sample_bernoulli
report_calls_to_sample_bernoulli = false;

test_rbm_w = a4_rand([100, 256], 0) * 2 - 1;
small_test_rbm_w = a4_rand([10, 256], 0) * 2 - 1;

temp = extract_mini_batch(data_sets.training, 1, 1);
data_1_case = sample_bernoulli(temp.inputs);
temp = extract_mini_batch(data_sets.training, 100, 10);
data_10_cases = sample_bernoulli(temp.inputs);
temp = extract_mini_batch(data_sets.training, 200, 37);
data_37_cases = sample_bernoulli(temp.inputs);

test_hidden_state_1_case = sample_bernoulli(a4_rand([100, 1], 0));
test_hidden_state_10_cases = sample_bernoulli(a4_rand([100, 10], 1));
test_hidden_state_37_cases = sample_bernoulli(a4_rand([100, 37], 2));

report_calls_to_sample_bernoulli = true;

clear temp;
