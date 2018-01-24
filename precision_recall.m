function [precision, recall]=precision_recall(predicted_class,true_class)
labels=unique(true_class,'stable');
    class_precision=zeros(length(labels),1);
    class_recall=zeros(length(labels),1);
    for i=1:length(labels)
        classInd_i=find(true_class==labels(i));
        TPi=length(find(predicted_class(classInd_i(1):classInd_i(end))==labels(i)));
        FPi=length(find(predicted_class==labels(i)))-TPi;
        FNi=length(find(predicted_class(classInd_i(1):classInd_i(end))~=labels(i)));   
        class_precision(i)=TPi/(TPi+FPi);
        class_recall(i)=TPi/(TPi+FNi);
    end
    class_precision(isnan(class_precision))=0;
    precision=mean(class_precision);
    recall=mean(class_recall);
end
