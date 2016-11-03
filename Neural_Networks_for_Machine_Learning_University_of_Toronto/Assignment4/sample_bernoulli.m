function binary = sample_bernoulli(probabilities)
    global report_calls_to_sample_bernoulli
    if report_calls_to_sample_bernoulli, 
        fprintf('sample_bernoulli() was called with a matrix of size %d by %d. ', size(probabilities, 1), size(probabilities, 2));
    end
    seed = sum(probabilities(:));
    binary = +(probabilities > a4_rand(size(probabilities), seed)); % the "+" is to avoid the "logical" data type, which just confuses things.
end

