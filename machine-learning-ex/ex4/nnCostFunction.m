function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
%+1 to account for the bias unit
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X_og = X;
X = [ones(m,1) X]; %add bias unit to x
h_1 = sigmoid(Theta1*X'); %25*5000
h_1 = [ones(m,1) h_1'];%5000*26, add bias unit
h_2 = sigmoid(Theta2*h_1'); %10*5000

y_temp = zeros(m,num_labels); %5000*10
for i=1:length(y) %loop over each y obs
    y_value = y(i); %get the value of y at a certain obs
    %y-value will be used to insert the 1 at that index
    y_tem = zeros(1,num_labels); %create a vector of zeroes (length 10)
    y_tem(y_value) = 1;
    y_temp(i,:) = y_tem'; %add in row of y vectors at a time
end

sum_k = 0;
for i=1:m %loop over no. of examples
    y_row = y_temp(i,:); %vector of length 10, 1*10
    h = h_2(:,i); %10*1
    sum_j = y_row*log(h)+(ones(size(y_row))-y_row)*log(ones(size(h))-h);
    sum_k = sum_k + sum_j;
end

J = (-1/m)*sum_k;

theta1_sq = Theta1.^2;
theta1_sq(:,1) = 0;
theta2_sq = Theta2.^2;
theta2_sq(:,1) = 0;
J = J+(lambda/(2*m))*(sum(theta1_sq,'all') + sum(theta2_sq,'all'));

h_2 = h_2'; %5000*10

for t=1:m %iterate over the example
    %for k=1:num_labels %iterate over each class aka 'output unit'     
    delta_3 = h_2(t,:) - y_temp(t,:); %1*10, corresponding to each k
    z_2 = Theta1*X(t,:)'; %z_2 = 25*1
    %Theta1=25*401, X'=401*1, z_2 = 25*1
    
    z_prime_2 = sigmoidGradient(z_2); %25*1    
    delta_2 = (Theta2(:,2:end)'*delta_3').*z_prime_2; %25*1 %remove d(0)
    %Theta2'=26*10, delta3'=10*1
    
    a_2 = sigmoid(z_2); %25*1, dim z_2=a_2
    a_2 = [ones(size(a_2,2)); a_2]; %add bias unit %26*1
    
    Theta2_grad = Theta2_grad + (a_2*delta_3)'; %26*10 %represents the D of every node
    Theta1_grad = Theta1_grad + (X(t,:)'*delta_2')'; %X'=401*1, delta_2'=1*25, D_1 = 401*25
end        

%Theta1_grad = (1/m)*Theta1_grad; %theta1 = 25*401
%Theta2_grad = (1/m)*Theta2_grad; %theta2 = 10*26

Theta1_grad(:,1) = (1/m)*Theta1_grad(:,1); %no regularisation for first column
Theta1_grad(:,2:end) = (1/m)*Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end)); 

Theta2_grad(:,1) = (1/m)*Theta2_grad(:,1); %no regularisation for first column
Theta2_grad(:,2:end) = (1/m)*Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end)); 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
