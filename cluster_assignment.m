function assignment=cluster_assignment(model,test_feat)

y=[];
for i=1:size(model.MU,2)
    pred=logmvnpdf(test_feat',model.MU(:,i)',diag(model.S(:,i))+0.0005);
    y=[y,pred];
end
yy=[];
for i=1:model.N_cluster:size(model.MU,2)
    yy=[yy,sum(y(:,i:i+model.N_cluster-1),2)];   
end

[~,Ind]=max(yy,[],2);
assignment=Ind;

end
