%clc;
clear;

% Load dataset
path='dataset/';
load ([path,'SUN.mat']);

train_class=size(trainClassLabels,1); % train class
test_class=size(testClassLabels,1);   % test class
%
test_feat=double(test_feat);
classAttributes=classAttributes';

attribute_dim=size(classAttributes,2);
[d,Ns]=size(train_feat);
A=classAttributes(trainClassLabels,:)';  
%
%================================================
K_trtr = kernelPoly(A',A',2);
K_trte = kernelPoly(A',classAttributes(testClassLabels,:),2);
%================================================


mu_cap=zeros(d,train_class);
sigma_s=zeros(d,train_class);

for i=1:train_class
    temp=trainClassLabels(i);
    class_feat=train_feat(:,train_labels==temp);
    MU=mean(class_feat,2);
    S=var(class_feat');
    mu_cap(:,i)=MU;
    sigma_s(:,i)=S;                
end

mu_unk=zeros(d,test_class);
sigma_unk=zeros(d,test_class);

% Hyperparameter lamda1 & lamda2
lamda1=0.1;lamda2=100000000; reg=0.05;

alpha_mu = (K_trtr+lamda1*eye(train_class))\mu_cap(:,:)';
mu_unk(:,:)=alpha_mu'*K_trte;


logsigmaS=log(sigma_s(:,:)+.001); % 0.1 added for stability
alpha = (K_trtr+lamda2*eye(train_class))\logsigmaS';
sigma_unk(:,:)=exp(alpha'*K_trte);

PComponents=ones(1,test_class)/test_class;
opt.PComponents=PComponents;
opt.testClassLabels=testClassLabels;
opt.test_labels=test_labels;
opt.regulariser=reg; % for stability
opt.mu_unk=mu_unk;
opt.sigma_unk=sigma_unk;

% Inductive setting
[Accuracy1]=Inductive_setting(test_feat,opt);
%  Transductive Setting
[Accuracy2]=Transductive_setting(test_feat,opt);
result=[Accuracy1,Accuracy2];

disp(['Inductive Accuracy = ',num2str(Accuracy1), '%  Transductive Accuracy = ', num2str(Accuracy2),'%'])

