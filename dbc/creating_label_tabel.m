function label_tabel=creating_label_tabel(train_data,train_label,number_of_hypothesis)
% Initialize by PCA

[signals,PC,V] = pca2(train_data);
label_tabel=(signals(1:number_of_hypothesis,:)>=0);

