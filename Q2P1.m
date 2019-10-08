% Expected risk minimization with 2 classes
clear all, close all,

n = 2; % number of feature dimensions
N = 400; % number of iid samples
mu(:,1) = [0;0]; mu(:,2) = [3;3];
Sigma(:,:,1) = [1 0;0 1]; Sigma(:,:,2) = [1 0;0 1];
p = [0.5,0.5]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

discriminantFunction = -0.5*(-2*((mu(:,2).')-(mu(:,1).'))*x+((mu(:,2).')*mu(:,2)-(mu(:,1).')*mu(:,1)))+log(p(2)/p(1));
decision = (discriminantFunction >= 0);

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0);
p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1);
no_of_errors = length(ind10)+length(ind01);%no.of errors
p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
p2_error = no_of_errors/N;% probability of error, empirically estimated

figure(1), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantFunctionGridValues = -0.5*[-2*(mu(:,2).'-mu(:,1).')*[h(:)';v(:)']+(mu(:,2).'*mu(:,2)-mu(:,1).'*mu(:,1))]+log(p(2)/p(1));
minDSGV = min(discriminantFunctionGridValues);
maxDSGV = max(discriminantFunctionGridValues);
discriminantFunctionGrid = reshape(discriminantFunctionGridValues,91,101);
figure(1), contour(horizontalGrid,verticalGrid,discriminantFunctionGrid,[0,0]); % decision boundary (contour at level 0)
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Decision Boundary' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 


