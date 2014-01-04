% Demo script to use "Add samples by categorical and exemplar attributes"
%
%  Written by Jonghyun Choi @ UMIACS UMD
%  Last updated @ 2013.11.22

clear;

nb     = 400; % size of attribute space
nClass = 1000; % number of total classes
cVal   = 0.1; % SVM parameter (in attribute space)

% nAddCategorical = [10 10]; % number of samples to be added by categorical attributes
% nAddExemplar    = [20 20]; % number of samples to be added by exemplar attributes
nAddCategorical = 20; % number of samples to be added by categorical attributes
nAddExemplar    = 30; % number of samples to be added by exemplar attributes

% date Load 
load( './data/lowlevel_feat.mat' ); % visual features
load('./data/dbc_models2new.mat'); % give: modelDBC

% % Or you can learn the Attribute space by the following with a labeled data downloaded from the internet
% [ modelDBC ] = learnDBC_model_adding( '.', org_lauxData, org_lauxData_oLabel, nb );

%% Baseline (Classification with init. set)
% learn SVM with the initial set
fprintf(1,'Train SVMs learned with Init. feature set\n');
aa = [ tr_labelSet10 ]; bb = [ tr_featSet10 ];
for i = setInt
    fprintf( 1, '[%d/%d]', i, setInt(end) );
    lset = -1*ones(size(aa)); lset(aa==i)=1; wp = 1/numel(lset==1); wn = 1/numel(lset==-1);
    svmORG{i} = train( lset', sparse(double(vl_homkermap(bb,3)')), ['-s 3 -c 1000 -B 1 -w1 ' num2str(wp) ' -w-1 ' num2str(wn) ' -q'] );
end
fprintf(1,'\n');

% Test with SVMs learned with the initial set
apsOrg = [];
fprintf(1,'Test with SVMs learned with Init. feature set\n');
for j = 1:numel(setInt), i = setInt(j);
    fprintf( 1, '[%d/%d]', i, setInt(end) );
    sm = svmORG{i};
    prob = sm.Label(1) * sm.w * [vl_homkermap(ts_featSet,3); ones(1,size(ts_featSet,2))];
    [rec, prec, th, ap] = precisionRecall(prob, double(ts_labelSet' == i));
    apsOrg(j) = ap;
end
fprintf(1,'\n');
apsOrg'
mean(apsOrg)


%% adding by Categorical and Exemplar Attributes
% initial training set
trSet.orgFeat = tr_featSet10;
trSet.label = tr_labelSet10;
% unlabeled data pool
uData.orgFeat = [ ul_featSet ] ;
uData.label = [ ul_labelSet ];

[ aSet, aLabel, oLabel ] = addByGnE( setInt, trSet, uData, modelDBC, cVal, nAddCategorical, nAddExemplar );

% learn SVM with AUG set
fprintf(1,'Train SVMs with aug feature set\n');
aa = [tr_labelSet10 aLabel]; bb=[tr_featSet10 aSet];
for i = setInt
    fprintf( 1, '[%d/%d]', i, setInt(end) );
    lset = -1*ones(size(aa)); lset(aa==i)=1; wp = 1/numel(lset==1); wn = 1/numel(lset==-1);
    svmAUG{i} = train( lset', sparse(double(vl_homkermap(bb,3)')), ['-s 3 -c 1000 -B 1 -w1 ' num2str(wp) ' -w-1 ' num2str(wn) ' -q'] );
end
fprintf(1,'\n');

% Test with updated SVMs
apsAug = [];
fprintf(1,'test with SVMs learned with aug feature set\n');
for j = 1:numel(setInt), i = setInt(j);
    fprintf( 1, '[%d/%d]', i, setInt(end) );
    sm = svmAUG{i};
    prob = sm.Label(1) * sm.w * [vl_homkermap(ts_featSet,3); ones(1,size(ts_featSet,2))];
    [rec, prec, th, ap] = precisionRecall(prob, double(ts_labelSet' == i));
    apsAug(j) = ap;
end
fprintf(1,'\n');
apsAug'
mean(apsAug)
