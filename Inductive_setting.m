function [Precision, Recall, Accuracy]=Inductive_setting(test_feat,opt)

y=[];
for i=1:size(opt.mu_unk,2)
    pred_prob=zeros(size(opt.test_labels,1),size(opt.mu_unk,3));
    for j=1:size(opt.mu_unk,3)       
        pred=logmvnpdf(test_feat',opt.mu_unk(:,i,j)',diag(opt.sigma_unk(:,i,j)+opt.regulariser));
        pred_prob(:,j)=pred;
    end

temp=sum(pred_prob,2);
y=[y,temp];
end
Nt=size(test_feat,2);
[~,Ind]=max(y,[],2);
index=opt.testClassLabels(Ind);
op=find((index-opt.test_labels)==0);
[Precision, Recall]=precision_recall(index,opt.test_labels);
Accuracy=(length(op)/Nt)*100;

end
