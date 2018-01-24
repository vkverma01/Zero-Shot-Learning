function [Precision, Recall, Accuracy]=Transductive_setting(test_feat,opt)

test_class=size(opt.mu_unk,2);
N_cluster=size(opt.mu_unk,3);
sigma_unkvl=zeros(size(test_feat,1),test_class*N_cluster);
mu_unkvl=zeros(size(test_feat,1),test_class*N_cluster);
it=1;
for i=1:test_class
    sigma_unkvl(:,it:it+N_cluster-1)=opt.sigma_unk(:,i,:);
    mu_unkvl(:,it:it+N_cluster-1)=opt.mu_unk(:,i,:); 
    it=it+N_cluster;
end

[model.MU,model.S,model.PI,~, POSTERIORS] = vl_gmm(test_feat,test_class*N_cluster, ...
                'initialization','custom', ...
                'InitMeans',mu_unkvl, ...
                'InitCovariances',sigma_unkvl, ...
                 'InitPriors',opt.PComponents, 'CovarianceBound', opt.regulariser, 'MaxNumIterations', 1000);


model.N_cluster=N_cluster;
clusterX=zeros(1,6180);
for i=1:test_class*N_cluster
   tmp=(POSTERIORS(i,:)>=0.5); 
   clusterX(tmp)=i;    
end

Nt=size(test_feat,2);
clusterX = cluster_assignment(model,test_feat);
index=opt.testClassLabels(clusterX);
op=find((index-opt.test_labels)==0);
Accuracy=(length(op)/Nt)*100;

[Precision, Recall]=precision_recall(index,opt.test_labels);  

end
