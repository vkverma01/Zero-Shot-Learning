function [Accuracy]=Inductive_setting(test_feat,opt)

y=[];
% opt.regulariser=0.1;
for i=1:size(opt.mu_unk,2)
    pred=logmvnpdf(test_feat',opt.mu_unk(:,i)',diag(opt.sigma_unk(:,i)+opt.regulariser));   
    y=[y,pred];
end
Nt=size(test_feat,2);
[~,Ind]=max(y,[],2);
index=opt.testClassLabels(Ind);
op=find((index-opt.test_labels)==0);
% [Precision, Recall]=precision_recall(index,opt.test_labels);
Accuracy=(length(op)/Nt)*100;

end
