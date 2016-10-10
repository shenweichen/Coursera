%% Plots information about a perceptron classifier on a 2-dimensional dataset.
function plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history)
%%
% The top-left plot shows the dataset and the classification boundary given by
% the weights of the perceptron. The negative examples are shown as circles
% while the positive examples are shown as squares. If an example is colored
% green then it means that the example has been correctly classified by the
% provided weights. If it is colored red then it has been incorrectly classified.
% The top-right plot shows the number of mistakes the perceptron algorithm has
% made in each iteration so far.
% The bottom-left plot shows the distance to some generously feasible weight
% vector if one has been provided (note, there can be an infinite number of these).
% Points that the classifier has made a mistake on are shown in red,
% while points that are correctly classified are shown in green.
% The goal is for all of the points to be green (if it is possible to do so).
% Inputs:
%   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
%       num_neg_examples is the number of examples for the negative class.
%   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
%       num_pos_examples is the number of examples for the positive class.
%   mistakes0 - A vector containing the indices of the datapoints from class 0 incorrectly
%       classified by the perceptron. This is a subset of neg_examples.
%   mistakes1 - A vector containing the indices of the datapoints from class 1 incorrectly
%       classified by the perceptron. This is a subset of pos_examples.
%   num_err_history - A vector containing the number of mistakes for each
%       iteration of learning so far.
%   w - A 3-dimensional vector corresponding to the current weights of the
%       perceptron. The last element is the bias.
%   w_dist_history - A vector containing the L2-distance to a generously
%       feasible weight vector for each iteration of learning so far.
%       Empty if one has not been provided.
%%
f = figure(1);
clf(f);

neg_correct_ind = setdiff(1:size(neg_examples,1),mistakes0);
pos_correct_ind = setdiff(1:size(pos_examples,1),mistakes1);

subplot(2,2,1);
hold on;
if (~isempty(neg_examples))
	plot(neg_examples(neg_correct_ind,1),neg_examples(neg_correct_ind,2),'og','markersize',20);
end
if (~isempty(pos_examples))
	plot(pos_examples(pos_correct_ind,1),pos_examples(pos_correct_ind,2),'sg','markersize',20);
end
if (size(mistakes0,1) > 0)
	plot(neg_examples(mistakes0,1),neg_examples(mistakes0,2),'or','markersize',20);
end
if (size(mistakes1,1) > 0)
	plot(pos_examples(mistakes1,1),pos_examples(mistakes1,2),'sr','markersize',20);
end
title('Classifier');

%In order to plot the decision line, we just need to get two points.
plot([-5,5],[(-w(end)+5*w(1))/w(2),(-w(end)-5*w(1))/w(2)],'k')
xlim([-1,1]);
ylim([-1,1]);
hold off;

subplot(2,2,2);
plot(0:length(num_err_history)-1,num_err_history);
xlim([-1,max(15,length(num_err_history))]);
ylim([0,size(neg_examples,1)+size(pos_examples,1)+1]);
title('Number of errors');
xlabel('Iteration');
ylabel('Number of errors');

subplot(2,2,3);
plot(0:length(w_dist_history)-1,w_dist_history);
xlim([-1,max(15,length(num_err_history))]);
ylim([0,15]);
title('Distance')
xlabel('Iteration');
ylabel('Distance');