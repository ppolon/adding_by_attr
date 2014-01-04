function label_tabel_new=update_label_tabel(label_tabel,num_exampels_per_cat,num_examples_in_cat)
num_of_cat=length(num_examples_in_cat);
[m n]=size(label_tabel);

lambda=(num_exampels_per_cat/n);

label_tabel_new=zeros(size(label_tabel));

for i=1:num_of_cat
    offset=sum(num_examples_in_cat(1:i-1))+1;
    S0=sum((label_tabel(:,offset:offset+num_examples_in_cat(i)-1)==0),2);
    S1=sum((label_tabel(:,offset:offset+num_examples_in_cat(i)-1)==1),2);  
    sum_0_cat(:,offset:offset+num_examples_in_cat(i)-1)=repmat(S0,[1,num_examples_in_cat(i)]);
    sum_1_cat(:,offset:offset+num_examples_in_cat(i)-1)=repmat(S1,[1,num_examples_in_cat(i)]); 
end

sum_0_all=repmat(sum((label_tabel==0),2),[1,n]);
sum_1_all=repmat(sum((label_tabel==1),2),[1,n]);
gradian_0=(1-label_tabel).*(sum_1_cat-lambda*(sum_1_all-sum_1_cat));
gradian_1=(label_tabel).*(sum_0_cat-lambda*(sum_0_all-sum_0_cat));
flip_mask=(gradian_0+gradian_1)>0;
label_tabel_new=xor(flip_mask,label_tabel);