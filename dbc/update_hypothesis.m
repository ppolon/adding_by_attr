function new_hypothesis=update_hypothesis(hypothesis,data,num_of_cat)
[m n]=size(data);
%  label_tabel=(hypothesis'*[data; ones(1,n)])>0;
%  new_hypothesis=train_hypothesis(data,label_tabel);
 


num_examples_per_cat=n/num_of_cat;
 label_tabel=(hypothesis'*[data; ones(1,n)])>0;
 for i=1:num_of_cat
     L(:,i)=mean(label_tabel(:,(i-1)*num_examples_per_cat+1:i*num_examples_per_cat),2)>0.5;
     label_tabel(:,(i-1)*num_examples_per_cat+1:i*num_examples_per_cat)=repmat(L(:,i),1,num_examples_per_cat);
 end
 new_hypothesis=train_hypothesis(data,label_tabel);