function opttheta = minfuncSGD(funcobj, theta, data, options)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  data       -  stores data in n x numExamples tensor
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, defualts to 0.9


%%======================================================================
%% Setup
assert(all(isfield(options, {'epochs', 'alpha', 'minibatch'})), ...
        'some ooptions not defined');
if ~isfield(options, 'momentum')
    options.momentum = 0.9;
end

epochs = options.epochs
alpha = options.alpha;
minibatch = options.minibatch;
m = size(data, 2); %% set size

% Setup for momentum
mom = 0.5;
momIncrease = 20;
velocity = zeros(size(theta));

%%======================================================================
%% SGD loop
it = 0;
for i = 1:epochs
    rp = ranperm(m);
    for s = 1:minibatch:(m-minibatch+1)
        it = it + 1;
        
        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = options.momentum;
        end
        
        % get next randomly selected minibatch
        mb_data = data(:, rp(s:s+minibatch-1));
        
         % evaluate the objective function on the next minibatch
        [cost grad] = funobj(theta, mb_data);
        
         % Instructions: Add in the weighted velocity vector to the
        % gradient evaluated above scaled by the learning rate.
        % Then update the current weights theta according to the
        % sgd update rule
        
        velocity = mom*velocity + alpha*grad;
        theta = theta - volocity;
    end
    
    alpha = alpha/2.0;
    
end

opttheta = theta;