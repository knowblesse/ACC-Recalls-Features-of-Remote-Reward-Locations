
clear
addpath(genpath('E:\MClust-4.3\'));
addpath(genpath('E:\DataHigh1.1\DataHigh1.1\'));

%%

folderNames = {'P1353_15p', 'P1353_16p', 'P1353_17p', 'P1353_18p', 'P1958_24p', 'P1958_25p', 'P1958_27p'};
img = cell(numel(folderNames), 2, 3);

for rat = 1:numel(folderNames)
    for offset = [0 1]    
        for trgt = 1:3 
            img{rat, offset+1, trgt} = zeros(1, 49);
        end
    end
end           

for rat = 1:7%numel(folderNames)

fprintf('rat %d \n ', rat); 
        
folderPath = ['E:\New folder\' folderNames{rat}  '\']; 
data = CreateAllData(folderPath, []);
% data.data(data.dataIndex, :) = smoother(data.data(data.dataIndex, :), 150, 1);

%%
[timestamps, binData, binLoc, trial] = BinData(data, 20, 0);
binLoc = MapToRect(binLoc, trial, data);

[code, codeMap] = CoarseGrid(binLoc, rat);
choice = [data.trInfo.choice];
ramp = [data.trInfo.rampTrial];
reward = [data.trInfo.durTrial];
freeChoice = [data.trInfo.freeChoice];

%%

mtTrial = cell(1, length(codeMap));
mtAll = cell(1, length(codeMap));
for i = 1:length(codeMap)
    
    if(sum(code == i) < 400)
        continue
    end   
    
    mtrial = [];
    mt = [];
    for tr = data.trials
        ind  = find(trial == tr & code == i);
        
        if(sum(ind) > 10)            
            dt = mean(binData(:, ind), 2);
            mt = [mt dt];
            mtrial = [mtrial tr];
        end       
        
                
    end    
    mtAll{i} = mt; 
    mtTrial{i} = mtrial;
end

for i = 39%1:length(codeMap) 
    
    fprintf('%d ',i);
    
    if(ismember(i, [9 10 12 13 16 17 19 20 23 24 26 27 30 31 33 34 37 38 40 41 ]))
        continue;
    end
    
    for offset = [0]    
        for trgt = 1:1            

            switch trgt
                case 1
                    target = freeChoice(mtTrial{i} - offset);
%                     target( target > 0) = 1;
                case 2
                    target = reward(mtTrial{i} - offset);
                    
                case 3
                    target = ramp(mtTrial{i} - offset);
                    target( target > 0) = 1;
            end            
            
            posTrial = mtTrial{i}(target == 1);
            negTrial = mtTrial{i}(target == 0);    

            minNum = min(numel(posTrial), numel(negTrial));                
            
            
            a = {};
            
            acc_iter = zeros(1, 30);
            for iter = 1:30

                posSample = randsample(posTrial, minNum);
                negSample = randsample(negTrial, minNum);

                selectedAll = [posSample negSample];
                selectedAll = selectedAll(randperm(numel(selectedAll)));

                r = rand(size(selectedAll));
                trainIndices = selectedAll(r > 0.3);
                testIndices  = selectedAll(r <= 0.3); 
                
                ind = ismember(mtTrial{i}, trainIndices);
                
                selectedCells = 1:size(mtAll{i}); 
                
                % setdiff(, 1), [3 6 26 33 40]); 
                % [6 5 14 15 22 24 25 51 53]
                     
                % 3 13 16 22 26 39 43
                % 1 9 12 25 8 33
                
                tr_data = mtAll{i}(selectedCells, ind);
                tr_tar  = target(ind);
                
                ind = ismember(mtTrial{i}, testIndices);
               
                ts_data = mtAll{i}(selectedCells, ind);
                ts_tar  = target(ind); 
                
%               SVMStruct = fitcsvm(tr_data', tr_tar', 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
%               preds = predict(SVMStruct, ts_data');

                

                [B, FitInfo] = lassoglm(tr_data', tr_tar', 'binomial', 'Alpha', 1, 'CV', 2, 'DFmax', 30, 'Options', statset('UseParallel', true));
                ind = FitInfo.IndexMinDeviance;
                
                a{iter} = find(B(:, ind) ~= 0);
                                
%                 ts_tar = ts_tar(randperm(numel(ts_tar)));
                
                preds = glmval([FitInfo.Intercept(ind); B(:, ind)], ts_data', 'logit') > 0.5;
                acc_iter(iter) = sum(preds(:) == ts_tar(:))/ numel(ts_tar(:));  

            end
            
%  0.5365 0.5307 0.4755 0.4918    0.6088   0.6067  

%             U = a{1};
%             for j = 2:10
%                 U = intersect(U, a{j});
%             end            
%             
%           intersect(intersect(intersect(intersect(a{1}, a{2}), a{3}), a{4}), a{5})
            
            avgAcc = mean(acc_iter);           
            img{rat, offset+1, trgt}(i) = avgAcc;
        end
    end
end

fprintf('\n',i);

for offset = [0 1]    
    for trgt = 1:1 
        img{rat, offset+1, trgt} = flip(rot90(reshape(img{rat, offset+1, trgt}, 7, [])', 2), 2);
    end
end


% for offset = 0%[0 1]
%     
%     fprintf('offset %d \n ',offset);
%     
%     for trgt = 1:3
%         
%         fprintf('target %d \n ', trgt);
%         
%         switch trgt
%             case 1
%                 target = choice(data.trials - offset);
%             case 2
%                 target = reward(data.trials - offset);
%             case 3
%                 target = ramp(data.trials - offset);
%         end           
%         
%         acc = zeros(1, length(codeMap));
%         
%         for i = 1:length(codeMap)
%             
%            fprintf('%d ', i);
% 
%             if(ismember(i, [ 1 7 9 10 12 13 16 17 19 20 23 24 26 27 30 31 33 34]))
%                 continue;
%             end
% 
%             posTrial = find(target == 1);
%             negTrial = find(target == 0);    
% 
%             minNum = min(numel(posTrial), numel(negTrial)); 
%             
%                 
%             acc_iter = zeros(1, 7);
%             for iter = 1:7
% 
%             posSample = randsample(posTrial, minNum);
%             negSample = randsample(negTrial, minNum);
% 
%             selectedAll = [posSample negSample];
%             selectedAll = selectedAll(randperm(numel(selectedAll)));
% 
%             r = rand(size(selectedAll));
%             trainIndices = selectedAll(r > 0.3);
%             testIndices  = selectedAll(r <= 0.3);   
% 
%             tr_data = mtAll{i}(:, trainIndices);
%             tr_tar  = target(trainIndices);
% 
%             ts_data = mtAll{i}(:, testIndices);
%             ts_tar  = target(testIndices) ; 
% 
% %             SVMStruct = fitcsvm(tr_data', tr_tar', 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
% %             preds = predict(SVMStruct, ts_data');            
% %             
%             [B, FitInfo] = lassoglm(tr_data', tr_tar', 'binomial', 'Alpha', 0.9, 'CV', 4, 'DFmax', 15, 'Options', statset('UseParallel', true));
% %             nNonZeroF = sum(B ~= 0);
%             ind = FitInfo.IndexMinDeviance; 
% 
%             preds = glmval([FitInfo.Intercept(ind); B(:, ind)], ts_data', 'logit') > 0.5;
%             acc_iter(iter) = sum(preds(:) == ts_tar(:))/ numel(ts_tar(:));  
% 
%             end
%             acc(i) = mean(acc_iter);            
%         end
%         
%         fprintf('\n ');
% 
%         acc(7) = mean([acc(6) acc(14)]);
%         acc(1) = mean([acc(2) acc(8)]);
%         img{rat, offset+1, trgt} = flip(rot90(reshape(acc, 7, [])', 2), 2);
% %         figure
% %         imagesc(img{rat, offset+1, trgt})
% %         title([ folderNames{rat}, '-', label{trgt},'-', offsetLabel{offset+1} ])
% %         caxis([0 1])
%         
%     end
% end

% pause(5)


end

% save('.\AccuracyMap\AccuracyMapLasso200', 'img');

label = {'Choice', 'Reward', 'Ramp'};
offsetLabel = {'Current', 'Previous'};

mnACC = cell(2, 3);
for j = 1:1%size(img, 2)
   for k = 1:1%2:size(img, 3)
       mnACC{j, k} = zeros(7, 7);
       for i = 1:6%size(img, 1)
           mnACC{j, k} = mnACC{j, k} + img{i, j, k};
       end

       mnACC{j, k} = mnACC{j, k} / 6; 
       mnACC{j, k}(49) = mean([mnACC{j, k}(48)  mnACC{j, k}(42)]);
       mnACC{j, k}(7) = mean([mnACC{j, k}(6)  mnACC{j, k}(14)]);
       
   end
end       


for j = 1:size(img, 2)
   for k = 1:size(img, 3)
        figure
        mnACC{j, k}(mnACC{j, k} < 0.4) = 0.4;
        imagesc(mnACC{j, k});
        title([ label{k},'-', offsetLabel{j}])
        caxis([0.4 1])
        colorbar
   end
end

