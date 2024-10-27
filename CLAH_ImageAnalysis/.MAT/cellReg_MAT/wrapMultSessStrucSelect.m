function [multSessSegStruc] = wrapMultSessStrucSelect(varargin);

%% USAGE: [multSessSegStruc] = wrapMultSessStrucSelect(varargin);
% Start with multSessTuninStruc or create new, selecting folders to
% compile.
% NOTE: abort file open dialog to stop adding new sessions.


if nargin == 1
    multSessSegStruc = varargin{1};
    disp('Appending to previous multSessTuningStruc');
else
    multSessSegStruc = struct();
    disp('Creating new multSessTuningStruc');
end

if length(multSessSegStruc)>1
    numSess = length(multSessSegStruc);
else
    numSess = 0;
end

% loop through and select Caiman output segDict for each session
stillAdding = 1;
while stillAdding
    try
        [segDictName, path] = uigetfile('*.mat', 'Select session segDict file to add to multSessTuningStruc');
        cd(path);
        
        %try
            %goodSegName = uigetfile('*.mat', 'Select goodSeg file for this segDict');
            
            disp(['Adding ' segDictName ' to multSessTuningStruc']);
        
            load(segDictName);
            
            %segName = findLatestFilename('segDict');
            %load(goodSegName);
            
            if strfind(segDictName, '2P')
               C = seg2P.C2p;
               A = seg2P.A2p;
               d1 = seg2P.d12p;
               d2 = seg2P.d22p;
               try
                   pksCell = seg2P.pksCell;
               catch
               end
            end
            
            %[treadBehStruc] = procHen2pBehav('auto');
            load(findLatestFilename('treadBehStruc'));
            
%             numBins = 100 ; rayThresh = 0.05;
%             [goodSegPosPkStruc, circStatStruc] = wrapUnitTuning(C, treadBehStruc, numBins, rayThresh);
            
            % put other things to calculate here, e.g.
            % xxxxx
            try
                load(findLatestFilename('cueShiftStruc'));
            catch
                disp('no cueShiftStruc');
            end
            
            numSess = numSess + 1;
            slashInds = strfind(path, '/');
            multSessSegStruc(numSess).path = path;
            multSessSegStruc(numSess).mouseName = path(slashInds(end-3)+1:slashInds(end-2)-1);
            multSessSegStruc(numSess).dayName = path(slashInds(end-2)+1:slashInds(end-1)-1);
            multSessSegStruc(numSess).sessName = path(slashInds(end-1)+1:end-1);
            multSessSegStruc(numSess).segDictName = segDictName;
            %multSessSegStruc(numSess).goodSegName = goodSegName;
            %multSessTuningStruc(numSess).sessPath = foldername;
            multSessSegStruc(numSess).C = C;
            multSessSegStruc(numSess).A = A;
            multSessSegStruc(numSess).d1 = d1;
            multSessSegStruc(numSess).d2 = d2;
            
            try
                multSessSegStruc(numSess).pksCell = pksCell;
            catch
            end
            
            
            multSessSegStruc(numSess).treadBehStruc = treadBehStruc;
            
            try
            multSessSegStruc(numSess).cueShiftStruc = cueShiftStruc;
            catch
            end
            
            %multSessSegStruc(numSess).avgTiff = imread(findLatestFilename('_avCaChDs'));
            
            %multSessSegStruc(numSess).goodSeg = goodSeg; % includes greatSeg but not in's and ok's
            %multSessSegStruc(numSess).greatSeg = greatSeg;
            
%             try
%             multSessSegStruc(numSess).okSeg = okSeg;
%             multSessSegStruc(numSess).inSeg = inSeg;
%             catch
%                 disp('No okSegs or INs');
%             end
%             
%             multSessSegStruc(numSess).pksCell = pksCell;
%             multSessSegStruc(numSess).deconvC = deconvC;
%             multSessSegStruc(numSess).posRates = posRates;
%             multSessSegStruc(numSess).posDeconv = posDeconv;

%             multSessTuningStruc(numSess).goodSegPosPkStruc = goodSegPosPkStruc;
%             wellTunedInd = find(circStatStruc.uniform(:,1)<0.01);
%             multSessTuningStruc(numSess).placeCellStruc.goodRay = wellTunedInd;
            
            % multSessTuningStruc(numSess).xxxx
            
%         catch
%             disp(['Couldnt process ' foldername ' (probably some file missing']);
%         end
    catch
        stillAdding = 0;
        disp('Canceled folder selection so aborting.');
%         saveFilename = ['/Backup20TB/clay/DGdata/multSessStrucs' multSessSegStruc(1).mouseName '/' multSessSegStruc(1).dayName '/' multSessSegStruc(1).mouseName '_' multSessSegStruc(1).dayName '_multSessSegStruc_' date '.mat'];
%         save(saveFilename, 'multSessSegStruc');
    end
     cd ../;
end




