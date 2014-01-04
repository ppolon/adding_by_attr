function model = DBC_train_2(train_data,train_label,nbits, label_tabel, fn )
% train_data: MxN: (M dimensions, N trials)
% train_label: 1xN: N: number of samples

% clean by Jonghyun Choi

number_of_hypothesis=nbits;

uni_labels=unique(train_label);
num_of_cat=length(uni_labels);

num_exampels_per_cat=sum(train_label==1);
for i=1:num_of_cat
    num_examples_in_cat(i)=length(find(uni_labels(i)==train_label));
end

[m n]=size(train_data);

% label_tabel=creating_label_tabel(train_data,train_label,number_of_hypothesis);

for i=1:2
    fn2 = [ fn '_' num2str(i) '.mat' ];
    if numel(dir(fn2)) > 0, load( fn2 ); % give: 'label_tabel', 'hypothesis'
    else
        %% Learning hypothesis(splits)
        hypothesis=train_hypothesis(train_data,label_tabel);
        if i>1
            for j=1, hypothesis=update_hypothesis(hypothesis,train_data,num_of_cat); end
        end

        %% Producing binary features
        [m n]=size(train_data);

        binary_features_train = (hypothesis'*[train_data; ones(1,n)])>0;

        % update binary labels
        label_tabel=binary_features_train;
        for j=1:10
            label_tabel=update_label_tabel(label_tabel,num_exampels_per_cat,num_examples_in_cat); 
        end
        save( fn2, 'label_tabel', 'hypothesis' );
    end
end

model.hypothesis=hypothesis;
model.nbits=nbits;