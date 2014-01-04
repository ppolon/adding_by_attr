function hypothesis=train_hypothesis(train_data,label_tabel)
%addpath('./liblinear-1.7/');
%addpath('./liblinear-1.7/matlab/');
[m n]=size(label_tabel);
number_of_hypothesis=m;
low_dim=length(train_data(:,1));
for i=1:number_of_hypothesis
    i
%     rand_mask{i}=repmat(rand(low_dim,1)>0.5,1,n);
%     train_data=train_data.*rand_mask;
    %% L2-svm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    train_label=label_tabel(i,:);
        pos_train_idx=find(train_label==1);
        neg_train_idx=find(train_label~=1);
%         num_of_negative_train=255;
%         neg_train_idx=floor(neg_train_idx(1:(length(neg_train_idx)/num_of_negative_train):end));
        posneg_train_data=double([train_data(:,pos_train_idx) train_data(:,neg_train_idx)]); % binarized verion of classemes
        posneg_train_label=[ones(length(pos_train_idx),1); -ones(length(neg_train_idx),1)];
        N=length(posneg_train_label);
        Np=sum(posneg_train_label==1);
        Nn=sum(posneg_train_label==-1);
    opt1 = [' -B 1 -c ' num2str(1) ' -s 1 -w-1 ' num2str((1/Nn)) ' -w1 ' num2str(1/Np)];
    %opt1 = ['-B 1 -c ' num2str(1000000) ' -s 6 -w-1 ' num2str((Np/N)) ' -w1 ' num2str(Nn/N)];
    %tic;
    model = train(posneg_train_label,sparse(posneg_train_data),opt1,'col');
    %time_learning_L2(i)=toc;
    
    hypothesis(:,i)=model.w';
    %b_L2(i)=model.w(end);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end