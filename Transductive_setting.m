function [Accuracy]=Transductive_setting(test_feat,opt)

test_class=size(opt.mu_unk,2);
N_cluster=1;
Nt=size(test_feat,2);
opt.regulariser=.4;
[model.MU,model.S,model.PI,~, ~] = vl_gmm(test_feat,test_class, ...
                'initialization','custom', ...
                'InitMeans',opt.mu_unk, ...
                'InitCovariances',opt.sigma_unk, ...
                 'InitPriors',opt.PComponents, 'CovarianceBound', opt.regulariser, 'MaxNumIterations', 1000);

model.N_cluster=N_cluster;

y=[];
for i=1:size(model.MU,2)
    pred=logmvnpdf(test_feat',model.MU(:,i)',diag(model.S(:,i))+0.05);
    y=[y,pred];
end

[~,clusterX]=max(y,[],2);
index=opt.testClassLabels(clusterX);
op=find((index-opt.test_labels)==0);
Accuracy=(length(op)/Nt)*100;
end
