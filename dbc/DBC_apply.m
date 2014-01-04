function H=DBC_apply(data,model)

[m n]=size(data);

H=(model.hypothesis'*[data; ones(1,n)])>0;