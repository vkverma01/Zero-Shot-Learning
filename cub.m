clc; 
clear;
% Load dataset
load ('zsl_dataset/cub200.mat');

train_class=size(trainClassLabels,1); % train class
test_class=size(testClassLabels,1);   % test class
test_feat=double(test_feat);

attribute_dim=size(classAttributes,2);
[d,Ns]=size(train_feat);
A=classAttributes(trainClassLabels,:)';  

%================================================
K_trtr = kernelPoly(A',A',2);
K_trte = kernelPoly(A',classAttributes(testClassLabels,:),2);
%================================================

N_cluster=1; % single gaussian
mu_cap=zeros(d,train_class);
sigma_s=zeros(d,train_class);
Pi=zeros(train_class);

for i=1:train_class
    temp=trainClassLabels(i);
    class_feat=train_feat(:,train_labels==temp);
    [MU,S,PI] = vl_gmm(class_feat,N_cluster); 
    mu_cap(:,i)=MU;
    sigma_s(:,i)=S;
    Pi(i)=PI;                
end

mu_unk=zeros(d,test_class);
sigma_unk=zeros(d,test_class);

% Hyperparameter lamda1 & lamda2
lamda1=5e8;
lamda2=1.5e9;

alpha_mu = (K_trtr+lamda1*eye(train_class))\mu_cap(:,:)';
mu_unk(:,:)=alpha_mu'*K_trte;

% logsigmaS=log(sigma_s_gmm(:,:,i)+0.1);
% lamda2=10000;
% V=logsigmaS*A'/(A*A'+lamda2*eye(attribute_dim));
% sigma_unk=exp(V*classAttributes(testClassLabels,:)');

logsigmaS=log(sigma_s(:,:)+0.1); % 0.1 added for stability
alpha = (K_trtr+lamda2*eye(train_class))\logsigmaS';
sigma_unk(:,:)=exp(alpha'*K_trte);

PComponents=ones(1,test_class*N_cluster)/test_class;
opt.PComponents=PComponents;
opt.testClassLabels=testClassLabels;
opt.test_labels=test_labels;
opt.regulariser=0.4;
opt.mu_unk=mu_unk;
opt.sigma_unk=sigma_unk;

% Inductive setting
[Precision1, Recall1, Accuracy1]=Inductive_setting(test_feat,opt);
disp(['Inductive_setting:: Precision= ',num2str(Precision1), ' & Recall= ',num2str(Recall1), ' & Accuracy= ',num2str(Accuracy1)])

% Transductive Setting
[Precision, Recall, Accuracy]=Transductive_setting(test_feat,opt);
disp(['Transductive_setting:: Precision= ',num2str(Precision), ' & Recall= ',num2str(Recall), ' & Accuracy= ',num2str(Accuracy)])


