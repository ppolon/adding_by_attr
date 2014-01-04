function model = DBC_train(train_data,train_label,nbits, opt1)
% Learning DBC hyperplanes
%  model = DBC_train(train_data, train_label, nbits, opt)
% Input: 
%  train_data: MxN: (M: number of dimensions, N: number of samples)
%  train_label: 1xN: N: number of samples
%  nbits: number of DBC hyperplanes
%  opt: SVM options (default: '-B 1 -c 1 -s 1')
% Output:
%  model: 
%
% Written by M.

% clean by Jonghyun Choi

if ~exist( 'opt1' ), opt1 = []; end
number_of_hypothesis=nbits;

uni_labels=unique(train_label);
num_of_cat=length(uni_labels);

num_exampels_per_cat=sum(train_label==1);
for i=1:num_of_cat
    num_examples_in_cat(i)=length(find(uni_labels(i)==train_label));
end

[m n]=size(train_data);

label_tabel=creating_label_tabel(train_data,train_label,number_of_hypothesis);

for i=1:2
    %% Learning hypothesis(splits)
    hypothesis=train_hypothesis(train_data,label_tabel, opt1);
    if i>1
        for j=1, hypothesis=update_hypothesis(hypothesis,train_data,num_of_cat, opt1); end
    end

    %% Producing binary features
    [m n]=size(train_data);

    binary_features_train = (hypothesis'*[train_data; ones(1,n)])>0;

    % update binary labels
    label_tabel=binary_features_train;
    for j=1:10
        label_tabel=update_label_tabel(label_tabel,num_exampels_per_cat,num_examples_in_cat); 
    end

end

model.hypothesis=hypothesis;
model.nbits=nbits;