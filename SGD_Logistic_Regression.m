%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% stochastic gradient descent for logistic regression exercise:
% trainig set includes 699 samples, where X contains 9-dimension feature
% vectors and y contains associated class labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use logististic regression model: 
%       p(y=1|x,beta)=1/(1+exp(beta*x'))=h(beta,x), 
%       p(y=-1|x,beta)=exp(beta*x')/(1+exp(beta*x'))=1-h(beta,x)
% Thus the logistic conditional probability: 
%       LCL=sum((1+yi)/2*log[h(beta,xi)]+(1-yi)/2*log[1-h(beta,xi)]) 
% The gradient of LCL can be calculated using the property: 
%       h(beta,x)'=h(beta,x)*[1-h(beta,x)]
% The gradiet, which is also the direction we pick for each iteration is: 
%       LCL'=[1/2+y/2-h(beta,x)]*x
% The descent rate can pick up value such as 1/k etc. that converges

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% simulation result:
%       beta= -0.2992
%              0.6308
%              0.2721
%              0.1161
%             -0.5759
%              0.4980
%             -0.5418
%              0.2857
%             -0.1993
% training error (misclassification fraction): 0.1316

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   author: Huiying Zhang
%   date: 03/04/2013


global feature_size
feature_size=9;
global sample_size
sample_size=699;

% load data
load('sys_7582_hw3.mat');

% initlize coefficient, sample index vector & beta 
k=1;
index=randperm(sample_size);
beta=zeros(feature_size,1);

% compute gradient and descent step at initial point
g=0;
for i=1:sample_size
    h=1/(exp(-beta'*X(:,i))+1);
    g=g+(1/2+y(i)/2-h)*X(:,1);
end
d=g;

% start SGD iterations   
while(d'*d>=1e-10 && k<sample_size )
    % sample and pick a direction
    i=index(k);
    d=-1/sample_size*g;
    
    % update beta along the direction with a rate of 1/sqrt(k)
    beta=beta+1/k^.5*d;
    
    % calculate gradient
    g=0;
    for i=1:k
        h=1/(exp(-beta'*X(:,i))+1);
        g=g-(1/2+y(i)/2-h)*X(:,i);
    end
    
    % update k for next iteration
    k=k+1;
    
end

% result of optimal parameter
sprintf('the optimal parameter is:')
disp(beta)

% calculate training error 
classification=0;
for i=1:sample_size
    % prediction label
    label=beta'*X(:,i);
    
    % number of correct classifications for y=1
    if label>0 && y(i)==1
        classification=classification+1;
    end
    % number of correct claasification for y=-1
    if label<0 && y(i)==-1
        classification=classification+1;
    end
    
end

% training error is misclassification number over total sample number
disp('training error is: ')
disp((sample_size-classification)/sample_size)