function describe_matrix(matrix)
    fprintf('Describing a matrix of size %d by %d. The mean of the elements is %f. The sum of the elements is %f\n', size(matrix, 1), size(matrix, 2), mean(matrix(:)), sum(matrix(:)))
end

