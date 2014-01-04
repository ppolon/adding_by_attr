function [ aSet, aLabel, oLabel ] = addByGnE( setInt, trSet, uData, modelDBC, cVal, kk_cs, kk_es )
% Find and add from unlabeled data by Categorical and Exemplar attributes
%
%  Written by Jonghyun Choi @ UMIACS UMD
%  Last updated @ 2013.11.15
%
%  This is a code of the following paper: 
%   Jonghyun Choi and Mohammad Rastegari and Ali Farhadi and Larry S.
%   Davis, "Adding Unlabeled Samples to Categories by Learned Attributes",
%   IEEE CVPR 2013
%
% input:
%  setInt: initial set category numbers (e.g. setInt = [ 1 2 3 4 ... 10 ])
%  trSet: training samples structure
%   .orgFeat (DVxNtr): visual features
%   .label   (1xNtr) : label
%  uData: unlabeled samples
%   .orgFeat (DVxNtr): visual features
%   .label   (1xNtr) : label
%  modelDBC: attribute space learned by auxiliary data
%  cVal: hyperparameter C for SVM training (LibLinear)
%  kk_cs (1xK): number of samples to be added by categorical attributes at each
%        iteration. (e.g., kk_c=[10 20 30] means it runs 3 iterations to add 60 samples in total)
%  kk_es (1xK): number of samples to be added by exemplar attributes at each
%        iteration. (e.g., kk_e=[10 20 30] means it runs 3 iterations to add 60 samples in total)
%
% output: 
%  aSet: added visual feature set
%  aLabel: obtained label by our algorithm
%  oLabel: original label (for evaluation purposes)
%
% Note: Mutual exclusion term is not implemented here yet.

addpath( './dbc/' );

trFeat  = trSet.orgFeat;
trLabel = trSet.label;

uFeat   = uData.orgFeat;
uLabel  = uData.label;
uDBC    = DBC_apply( vl_homkermap(uFeat,3), modelDBC );
idxSet  = 1:numel(uLabel);

if numel(kk_cs) ~= numel(kk_es), error( 'Dimensions of ''kk_cs'' and ''kk_es'' should be identical!' ); end

% init var.
newSet_feat  = trFeat;
newSet_label = trLabel;
aSet = []; aLabel = []; aIdx = [];

% block coordinate descent loop
for jh = 1:numel(kk_cs)
    fprintf(1,'iter: %d\n', jh);
    kk_c = kk_cs(jh);
    kk_e = kk_es(jh);
    
    newSet_feat  = [ newSet_feat aSet ];
    newSet_label = [ newSet_label aLabel ];
    trDBC   = DBC_apply( vl_homkermap(newSet_feat,3), modelDBC );
    
    % categorical
    for i = setInt
        lset = -1*ones(size([newSet_label ])); lset([newSet_label ]==i)=1; 
        wp = 1/sum(lset==1); wn = 1/sum(lset==-1);
        gSvm = train( lset', sparse(double([trDBC ]')), ['-s 3 -c ' num2str(cVal) ' -B 1 -w1 ' num2str(wp) ' -w-1 ' num2str(wn) ' -q'] );
        svmDBC{i}.wg = gSvm.Label(1)*gSvm.w(1:end-1)';
    end

    % exemplar by leave-one-out comparison
    for j = 1:numel(setInt), i = setInt(j);
        pI = find( newSet_label == i ); % positive index
        Winit = [];
        for jjj = 1:numel(pI), jj = setdiff( pI, pI(jjj) );
            lset = [ 1*ones(1,numel(pI)-1) -1*ones(1,sum(newSet_label~=i)) ]; 
            wp = 1/sum(lset==1); wn = 1/sum(lset==-1);

            eSvm = train( lset', sparse(double([trDBC(:,[jj find(newSet_label~=i)]) ]')), ['-s 3 -c ' num2str(cVal) ' -B 1 -w1 ' num2str(wp) ' -w-1 ' num2str(wn) ' -q'] );
            Winit(:,jjj) = eSvm.Label(1)*eSvm.w(1:end-1)';
        end
        svmDBC{i}.wl = Winit;
    end

    % adding
    aIdxHere = [];
    for j = 1:numel(setInt), i = setInt(j);
        wLoc = svmDBC{i}.wl;
        wGlo = svmDBC{i}.wg;

        d_loc = wLoc'*uDBC;
        d_glo = wGlo'*uDBC;

        % compute rank scores at each image
        clear rScores;
        [valGlo idxGlo]=sort(d_glo,'descend');
        rScores(1,idxGlo) = (1:size(uDBC,2));
        for nn = 1:size(wLoc,2)
            [valLoc idxLoc]=sort(d_loc(nn,:),'descend');
            rScores(nn+1, idxLoc) = 1:size(uDBC,2);
        end

        % making score
        rScoreSum = sum( rScores, 1 ); % compute score of rank sum
        rDscore = exemplar_score( rScores ); % compute score of rank difference by different metric

        % categorical
        kk = kk_c;
        [val idx]=sort(rScoreSum,'ascend');
        aSet = [aSet, uFeat(:,idx(1:kk))];
        aLabel = [aLabel, i*ones(1,numel(idx(1:kk)))];

        oLabel = uLabel(idx(1:kk));
        
        aIdx = [ aIdx idxSet(idx(1:kk)) ];
        aIdxHere = [ aIdxHere idx(1:kk) ];
        
        % exemplar
        kk = floor( kk_e / size(wLoc,2) );
        for gg = 1:kk
            for nn = 1:size(rDscore,1)
                [val idx]=sort(rDscore(nn,:),'descend');

                % discard duplicated images
                [ nIdx nIdxII ] = setdiff( idx, aIdxHere );
                idx = idx(sort(nIdxII,'ascend'));

                aSet = [aSet, uFeat(:,idx(1))];
                aLabel = [aLabel, i*ones(1,numel(idx(1)))];
                
                oLabel = [ oLabel uLabel(idx(1)) ];
                
                aIdx = [ aIdx idxSet(idx(1)) ];
                aIdxHere = [ aIdxHere idx(1) ];
            end            
        end
    end
    % remove added item from unlabeled sample pool
    uFeat(:,aIdxHere) = [];
    uLabel(:,aIdxHere) = [];
    uDBC(:,aIdxHere) = [];
    idxSet(:,aIdxHere) = [];
end



%-------
function rDscore = exemplar_score( rScores )

rDscore = bsxfun( @minus, 1./rScores(1,:), 1./rScores(2:11,:) ); % 0        
% rDscore = bsxfun( @rdivide, bsxfun( @minus, rScores(2:11,:), rScores(1,:) ), log(rScores(1,:)+1) ); % 1    
% rDscore = bsxfun( @rdivide, bsxfun( @minus, rScores(2:11,:), rScores(1,:) ), rScores(1,:) ); % 2
% rDscore = bsxfun( @rdivide, log(bsxfun( @minus, rScores(2:11,:), rScores(1,:) )+1), log(rScores(1,:)+1)); % 3
% rDscore = log(bsxfun( @minus, rScores(2:11,:), rScores(1,:) )+1); % 4
% rDscore = bsxfun( @minus, log(rScores(2:11,:)+1), log(rScores(1,:)+1) ); % 5
% rDscore = bsxfun( @rdivide, bsxfun( @minus, log(rScores(2:11,:)+1), log(rScores(1,:)+1) ), log(rScores(1,:)+1) ); % 6
