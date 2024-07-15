function [] = testing_cross_validation_for_paper()

% Function to assess the validity of statistical testing in Adam White's
% first machine learnig paper.
%   Set-up a classification problem under the null, perform k fold
% cross validation, with a lock-box, as per Adam's approach. Test a range
% of different statistical procedures to see whether we observe an
% inflated false positive (type I error) rate.
%
% Key property: the classification losses we see are more different across
% two models than they are across folds of the same model.
% This is the key property that breaks statistical inference. It means
% that true observed differences between the two models are extreme
% relative to the variability under the null.

%% Parameter settings
% param_paper are the settings used in Adam's paper
nOfObservations = 960; % 960[param_paper] Must be divisible by 8
nOfDataVariables = 18; % 18[param_paper]; Must be divisible by 18.
nOfModels = 2; % Num. of machine learning models
nOfReps = 800; % 800[param_paper] (old vals: 20, 60) Num. of repeated comparisons of
             % models, e.g. generating a p-value
nFolds = 4; % 4[param_paper] (Old vals: 12, 24, 4) should keep
        % nOfObservations/nFolds relatively large, otherwise the set of 
        % possible accuracy results for a particular set up of data is small
        % and granularity of classification loss is too coarse.
nPerms = 500; % 500[param_paper] (Old vals: 16, 24, 100) Number of resamplings
              % that perform for each permutation test.

% Following allow the degrees of freedom to be adjusted, to reflect
% correlations between folds.
downScale1 = 0.45;
downScale2 = 0.435;
downScale3 = 0.4;

% Can turn whether have a lockbox on and off
lockbox = true; % true[param_paper]

signal_amplitude = 0.25; % 0.25[param_paper] (Previously 0.2, 0.4) When set to 1,
            % has same amplitude as noise. If set to 0, there is no signal.
alpha = 0.05; % Significance threshold

%% Vectors to accumulate p-values

% Parametric tests, paired and independent t-tests
p_vals_param_paired = zeros(nOfReps,1);
p_vals_param_ind = zeros(nOfReps,1);

% Parametric, with adjustment of degrees of freedom
p_vals_param_adjusted_paired1 = zeros(nOfReps,1);
p_vals_param_adjusted_paired2 = zeros(nOfReps,1);
p_vals_param_adjusted_paired3 = zeros(nOfReps,1);

% Non-parametric: Wilcoxon (paired & independent)
p_vals_wilcoxon_paired = zeros(nOfReps,1);
p_vals_wilcoxon_ind = zeros(nOfReps,1);
% Non-parametric: Permutation (paired & independent)
p_vals_perm_paired = zeros(nOfReps,1);
p_vals_perm_ind = zeros(nOfReps,1);
               
%% Main loop
% Each iteration generates a p-value for every statistical procedure.
for k=1:nOfReps

    %Data struct to store classification loss across data splits.
    LossVector_folds = zeros(nFolds,nOfModels);
    LossVector_lockbox = zeros(nFolds,nOfModels);

    %Patterns to put into class 1 and 2 observations
    pattern_cl1 = normrnd(0,signal_amplitude,1,nOfDataVariables);
    pattern_cl2 = normrnd(0,signal_amplitude,1,nOfDataVariables);

    %Pattern in class 1 and 2, main data
    signal_class_1 = repmat(pattern_cl1,(nOfObservations/2),1);
    signal_class_2 = repmat(pattern_cl2,(nOfObservations/2),1);
    signal_main = cat(1,signal_class_1,signal_class_2);
    %Pattern in class 1 and 2 lockbox
    signal_class_1_LB = repmat(pattern_cl1,(nOfObservations/(nFolds*2)),1);
    signal_class_2_LB = repmat(pattern_cl2,(nOfObservations/(nFolds*2)),1);
    signal_LB = cat(1,signal_class_1_LB,signal_class_2_LB);

    %% Set-up data
    % Main data
    data = normrnd(0,1,nOfObservations,nOfDataVariables); %Add random fluctuations
    dataWithSignal = data + signal_main;
    
    % Lock-box
    lockbox_data = normrnd(0,1,(nOfObservations/nFolds),nOfDataVariables);  %Add random fluctuation
    lockbox_dataWithSignal = lockbox_data + signal_LB;

    %Set-up class labels
    class = strings([nOfObservations,1]);
    for i=1:(nOfObservations/2) 
        class(i,1)="class1";
        class((nOfObservations/2)+i,1)="class2";
    end
    lock_box_class = strings([(nOfObservations/nFolds),1]);
    for i=1:(nOfObservations/(nFolds*2))
        lock_box_class(i,1)="class1";
        lock_box_class((nOfObservations/(nFolds*2))+i,1)="class2";
    end

    %% Simulate null
    % Modelling null is tricky, since different classifiers applied to same
    % data (even if null) might exhibit a real difference in classification
    % performance. So, we simulate null by comparing the performance of the
    % same classifier applied to different variables in the data.

    % The way we remove a variable fromm the data is to just set it to zero for
    % all observations, whether in class 1 or 2. So, the difference between
    % classifiers really comes through feature selection.
    % Accordingly, start with data structs of zeros
    dataWithSignal4mdl_1 = zeros(nOfObservations,nOfDataVariables);
    dataWithSignal4mdl_2 = zeros(nOfObservations,nOfDataVariables);
    % Same for lock-box
    lockbox_dataWithSignal4mdl_1 = zeros(nOfObservations/nFolds,nOfDataVariables);
    lockbox_dataWithSignal4mdl_2 = zeros(nOfObservations/nFolds,nOfDataVariables);

    % 2/18 of the data is different between the 2 models. 
    EighteenthOfVars = nOfDataVariables/18;
    % Data for first model: last 18th of variables left as zeros.
    dataWithSignal4mdl_1(:,(1:(nOfDataVariables-EighteenthOfVars))) = dataWithSignal(:,(1:(nOfDataVariables-EighteenthOfVars)));
    % Data for second model: first 18th of variables left as zeros.
    dataWithSignal4mdl_2(:,((EighteenthOfVars+1):nOfDataVariables)) = dataWithSignal(:,((EighteenthOfVars+1):nOfDataVariables));
    % Similar for lockbox
    % First model
    lockbox_dataWithSignal4mdl_1(:,(1:(nOfDataVariables-EighteenthOfVars))) = lockbox_dataWithSignal(:,(1:(nOfDataVariables-EighteenthOfVars))); 
    % Second model
    lockbox_dataWithSignal4mdl_2(:,((EighteenthOfVars+1):nOfDataVariables)) = lockbox_dataWithSignal(:,((EighteenthOfVars+1):nOfDataVariables));

    %% Fit linear discriminant models and cross validate - same folds for both models
    % Need to use pseudo linear because zeros make covariance matrix
    % degenerate.
    cvp = cvpartition(class,'Kfold',nFolds);
    for j=1:nOfModels
    
        if j==1 % Iteration 1: one porportion of data set to zero
            cvMdl = fitcdiscr(dataWithSignal4mdl_1,class,'CVPartition',cvp,'discrimType','pseudoLinear');
            lockbox_dataWithSignal_current_model = lockbox_dataWithSignal4mdl_1;
        else % Iteration 2: different proportion of data set to zero
            cvMdl = fitcdiscr(dataWithSignal4mdl_2,class,'CVPartition',cvp,'discrimType','pseudoLinear');
            lockbox_dataWithSignal_current_model = lockbox_dataWithSignal4mdl_2;
        end
        
        % Calculate kfoldLoss and store classification loss for all folds.
        % Classification loss can be thought of as one minus accuracy.
        LossVector_folds(:,j) = kfoldLoss(cvMdl,'mode','individual');

        % Calculate lockbox loss for each fold
        for t=1:nFolds
            classifier_current_fold = cvMdl.Trained{t};
            LossVector_lockbox(t,j) = loss(classifier_current_fold,lockbox_dataWithSignal_current_model,lock_box_class);
        end
    end

    %% Swap between running statistical inference over tests
    % on lockbox and on folds
    if lockbox
        x=LossVector_lockbox(:,1);
        y=LossVector_lockbox(:,2);
    else
        x=LossVector_folds(:,1);
        y=LossVector_folds(:,2);
    end

    %% Classic two-sample t-tests. THIS IS A 2-SIDED TEST
    [h_paired,p_paired,ci,stats_paired] = ttest(x,y); %paired test
    p_vals_param_paired(k,1) = p_paired;
    dofs1 = stats_paired.df;

    [h_ind,p_ind,ci,stats_ind] = ttest2(x,y); %independent test
    p_vals_param_ind(k,1) = p_ind;
    dofs2 = stats_ind.df;


    %% Adjusted two-sample paired t-test, which is one-tailed
    % Allows the degrees of freedom to be adjusted, to reflect
    % correlations between folds.
    pval_adjusted_paired = Adjusted_paired_t_test(x,y,downScale1);
    p_vals_param_adjusted_paired1(k,1) = pval_adjusted_paired;
    pval_adjusted_paired = Adjusted_paired_t_test(x,y,downScale2);
    p_vals_param_adjusted_paired2(k,1) = pval_adjusted_paired;
    pval_adjusted_paired = Adjusted_paired_t_test(x,y,downScale3);
    p_vals_param_adjusted_paired3(k,1) = pval_adjusted_paired;

    %% Wilcoxon two-sample tests. THIS IS A 2-SIDED TEST
    p_paired2 = signrank(x,y); % paired test
    p_vals_wilcoxon_paired(k,1) = p_paired2;
    p_ind2 = ranksum(x,y); % independent test
    p_vals_wilcoxon_ind(k,1) = p_ind2;

    %% Permutation tests. THESE ARE 1-SIDED TESTS.
    % Two-sample paired permutation test
    p_vals_perm_paired(k,1) = coin_flip_paired_perm_test(x,y,nFolds);
    % Two-sample independent permutation test
    p_vals_perm_ind(k,1) =  shuffle_all_perm_test(x,y,nFolds);

    disp("Still in main loop: k is iteration number, nOfReps is number of repetitions");
    k
    nOfReps
end

%% Two-tailed permutation
p_vals_perm_paired_two_tailed = arrayfun(@(p) OnetoTwo_tailed(p),p_vals_perm_paired);
p_vals_perm_ind_two_tailed = arrayfun(@(p) OnetoTwo_tailed(p),p_vals_perm_ind);

% p_vals for adjusted t-test
% Make two-tailed
p_vals_param_adjusted_paired_twotailed_1 = arrayfun(@(p) OnetoTwo_tailed(p),p_vals_param_adjusted_paired1);
p_vals_param_adjusted_paired_twotailed_2 = arrayfun(@(p) OnetoTwo_tailed(p),p_vals_param_adjusted_paired2);
p_vals_param_adjusted_paired_twotailed_3 = arrayfun(@(p) OnetoTwo_tailed(p),p_vals_param_adjusted_paired3);

% Calc type-I error rates and plot as bar-chart
type_I_errors = zeros(1,9);
% t paired
signif_p_vals = p_vals_param_paired(p_vals_param_paired<alpha);
type_I_errors(1,1) = (length(signif_p_vals)/length(p_vals_param_paired))*100;
% t indep
signif_p_vals = p_vals_param_ind(p_vals_param_ind<alpha);
type_I_errors(1,2) = (length(signif_p_vals)/length(p_vals_param_ind))*100;

% t paired, adjusted, two-tailed
signif_p_vals = p_vals_param_adjusted_paired_twotailed_1(p_vals_param_adjusted_paired_twotailed_1<alpha);
type_I_errors(1,3) = (length(signif_p_vals)/length(p_vals_param_adjusted_paired_twotailed_1))*100;
signif_p_vals = p_vals_param_adjusted_paired_twotailed_2(p_vals_param_adjusted_paired_twotailed_2<alpha);
type_I_errors(1,4) = (length(signif_p_vals)/length(p_vals_param_adjusted_paired_twotailed_2))*100;
signif_p_vals = p_vals_param_adjusted_paired_twotailed_3(p_vals_param_adjusted_paired_twotailed_3<alpha);
type_I_errors(1,5) = (length(signif_p_vals)/length(p_vals_param_adjusted_paired_twotailed_3))*100;

% Wilcoxon paired
signif_p_vals = p_vals_wilcoxon_paired(p_vals_wilcoxon_paired<alpha);
type_I_errors(1,6) = (length(signif_p_vals)/length(p_vals_wilcoxon_paired))*100;
% Wilcoxon independent
signif_p_vals = p_vals_wilcoxon_ind(p_vals_wilcoxon_ind<alpha);
type_I_errors(1,7) = (length(signif_p_vals)/length(p_vals_wilcoxon_ind))*100;
% Permutation paired
signif_p_vals = p_vals_perm_paired_two_tailed(p_vals_perm_paired_two_tailed<alpha);
type_I_errors(1,8) = (length(signif_p_vals)/length(p_vals_perm_paired_two_tailed))*100;
% Permutation independent
signif_p_vals = p_vals_perm_ind_two_tailed(p_vals_perm_ind_two_tailed<alpha);
type_I_errors(1,9) = (length(signif_p_vals)/length(p_vals_perm_ind_two_tailed))*100;

%PERCENT BELOW .05 AS A BAR-CHART.
figure
labels = ["t-paired","t-indep","t-paired-bast1","t-paired-bast2","t-paired-bast3","Wilcoxon-paired","Wilcoxon-indep","Perm-paired","Perm-indep"];
bar(labels,type_I_errors);
title('Type-I Error Rates');
xlabel('Statistic');
ylabel('Percentage of false positives');

return;


function [p_from_t] = Adjusted_paired_t_test(x_in,y_in,down_scale)
% Reconstructed t-test, in which the degrees of freedom can be scaled.
% x_in and y_in are the two conditions being compared.

    % Work out degrees of freedom
    N = size(x_in,1);
    apparent_dof = N-1;
    effective_dof = apparent_dof * down_scale;

    if (effective_dof<=1)
        disp("ERROR: effective degrees of freedom are too small to calculate a p_value; see ££");
    end

    % Components of formula
    diff_xy = x_in(:) - y_in(:);
    mean_diff_xy = mean(diff_xy,1);
    diffFromMean = diff_xy(:) - mean_diff_xy;
    squared_diffFromMean = diffFromMean.^2;
    sum_squared_diffFromMean = sum(squared_diffFromMean);

    % Build t formula
    denominator = sqrt(sum_squared_diffFromMean/(apparent_dof));
    numerator = mean_diff_xy * sqrt(effective_dof+1);
    t_val = numerator / denominator;

    p_from_t = 1-tcdf(t_val,effective_dof); % ££; p-value from t-statistic
    return;
end


function [p_new] = OnetoTwo_tailed(p)

    if p<0.5
        p_twoTailed = 2*p;
    else
        p_twoTailed = 2*(1-p);
    end
    p_new = p_twoTailed;
    return;
end


function [p] = coin_flip_paired_perm_test(xx,yy,nIters)
% Params: x, y; return p-value
% Two-sample paired permutation test.
% Note, there are 16 alternatives under this procedure with 4 folds.
% However, because of monte-carlo sampling, this can generate a large
% range of p-values than just 16. This is because, just by chance
% the null distribution could contain a "disproportionate" number of
% instances at any particular point.
% It may be that it would be better doing an exact permutation test
% when there are few folds. In a sense, the Monte-Carlo sampling is
% adding uncertainty that does not have to be there. Although, clearly
% if we do enough Monte-Carlo resamplings the p-values one gets will
% be relatively sensible.

    true_observed_diff_of_means = mean(xx)-mean(yy);

    perm_dist_paird = zeros(nPerms,1);
    for mm=1:nPerms
        surrogate_xx = xx;
        surrogate_yy = yy;
        for nn=1:nIters
            %flip a coin
            fflip = false;
            rrnd = rand();
            if rrnd >0.5
                fflip = true;
            end
            % if flip, swap values between two surrogates
            if fflip
                ttemp=surrogate_xx(nn,1);
                surrogate_xx(nn,1) = surrogate_yy(nn,1);
                surrogate_yy(nn,1) = ttemp;
            end
        end
        perm_dist_paird(mm,1) = mean(surrogate_xx)-mean(surrogate_yy);

    end
    %perm_dist_paired
    perm_dist_paird_filtered = perm_dist_paird(perm_dist_paird>true_observed_diff_of_means);
    p = length(perm_dist_paird_filtered)/length(perm_dist_paird);
    return;
end


function [p] = shuffle_all_perm_test(xxx,yyy,nIters)
% Two-sample independent permutation test

    true_observed_diff_of_means2 = mean(xxx)-mean(yyy);

    perm_dist_ind = zeros(nPerms,1);
    xxx_yyy = cat(1,xxx,yyy);
    xxx_surrogate_indep = zeros(nIters,1);
    yyy_surrogate_indep = zeros(nIters,1);
    for mmm=1:nPerms
        xxx_yyy_shuffled = xxx_yyy(randperm(length(xxx_yyy)));
        for rrr=1:nIters
            xxx_surrogate_indep(rrr,1) = xxx_yyy_shuffled(rrr,1);
            yyy_surrogate_indep(rrr,1) = xxx_yyy_shuffled(nIters+rrr,1);
        end
        perm_dist_ind(mmm,1) = mean(xxx_surrogate_indep)-mean(yyy_surrogate_indep);
    end
    % perm_dist_indep
    perm_dist_ind_filtered = perm_dist_ind(perm_dist_ind>true_observed_diff_of_means2);
    p = length(perm_dist_ind_filtered)/length(perm_dist_ind);
    return;
end

end