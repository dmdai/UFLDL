function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    
% The input theta is a vector because minFunc only deal with vectors. In
% this step, we will convert theta to matrix format such that they follow
% the notation in the lecture notes.                                 
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

W2grad = zeros(size(W2));
b2grad = zeros(size(b2));
W1grad = zeros(size(W1));
b1grad = zeros(size(b1));

% Loss and gradient variables (your code needs to compute these values)
m = size(data, 2);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the loss for the Sparse Autoencoder and gradients
%                W1grad, W2grad, b1grad, b2grad
%
%  Hint: 1) data(:,i) is the i-th example
%        2) your computation of loss and gradients should match the size
%        above for loss, W1grad, W2grad, b1grad, b2grad

z2 = W1 * data + repmat(b1, 1, m);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2, 1, m);
a3 = z3;

Jcost = 0.5 * (1/m) * sum(sum((a3 - data).^2));
Jweight = 0.5 * lambda * (sum(W1(:).^2) + sum(W2(:).^2));
ro = 1/m * sum(a2, 2);
Jsparse = beta * sum(sparsityParam.*log(sparsityParam./ro) + ...
    (1-sparsityParam).*log((1-sparsityParam)./(1-ro)));

cost = Jcost + Jweight + Jsparse;

d3 = a3 - data;
sterm = beta * (-sparsityParam./ro + (1-sparsityParam)./(1-ro));
d2 = (W2'*d3 + repmat(sterm, 1, m)).*(a2.*(1-a2));

W1grad = W1grad + (1/m)*d2*data' + lambda*W1;
W2grad = W2grad + (1/m)*d3*a2' + lambda*W2;
b1grad = b1grad + (1/m)*sum(d2, 2);
b2grad = b2grad + (1/m)*sum(d3, 2);

%-------------------------------------------------------------------
% Convert weights and bias gradients to a compressed form
% This step will concatenate and flatten all your gradients to a vector
% which can be used in the optimization method.
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% We are giving you the sigmoid function, you may find this function
% useful in your computation of the loss and the gradients.
function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));
end

