function main_test()

% The path of the folder containing this script
BASE_PATH = 'Y:\PythonEEG\matlab'; % no trailing slash
DWDLL_PATH = [BASE_PATH filesep 'RequiredResources' filesep 'nsDWFile64.dll']

% The path of the folder containing all of the .DDF files to be converted
DDF_PATH = 'Y:\PythonEEG\ddfs';

% The path of the folder where converted .BIN files will be saved
% Set to '' to save where the DDF files were located
BIN_PATH = 'Y:\PythonEEG\bins';

% How many seconds of DDF file to read on each iteration. Large values
% preferred for speed, but too large will make the script run out of memory
TIME_BLOCK_INTERVAL = 1000; % Seconds

% Ignore DDF channels containing the word "filter", e.g. digitally filtered
% data
IGNORE_FILTER = true;

% How precise to store the data. Set the tradeoff between precision and
% filesize. See more options at:
% https://www.mathworks.com/help/matlab/ref/fwrite.html#buakf91-1-precision
SAVE_PRECISION = 'float32';

% Set the NeuroShare DLL path
ns_SetLibrary(DWDLL_PATH);
[~, nsLibraryInfo] = ns_GetLibraryInfo();
disp(nsLibraryInfo)

ddfs = dir([DDF_PATH filesep '*.ddf']);
disp([DDF_PATH filesep '*.ddf'])

temp = struct2table(ddfs);
ddfs_sorted = sortrows(temp, 'name');
ddfs_sorted = table2struct(ddfs_sorted);

for i = 1:numel(ddfs_sorted)
    disp(ddfs_sorted(i).name)
end

for i = 1:numel(ddfs_sorted)
    [~, hFile] = ns_OpenFile([ddfs_sorted(i).folder filesep ddfs_sorted(i).name]);
    [~, nsFileInfo] = ns_GetFileInfo(hFile);
    
    [~, fileStem, ~] = fileparts(ddfs_sorted(i).name);
    fileBin = strjoin([BIN_PATH filesep fileStem "_ColMajor" ".bin"], "")
    fileTxtInfo = strjoin([BIN_PATH filesep fileStem "_Meta" ".csv"], "")

    fileID = fopen(fileBin, 'w'); % Open BINary file to receive DDF data
    fileIDTxt = fopen(fileTxtInfo, 'w') % Open TXT file to write names of channels from DDF
    nChannels = nsFileInfo.EntityCount
    colCount = 1;
    
    fprintf(fileIDTxt, strjoin( ...
        ["Entity" "BinColumn" "Label" "ProbeInfo" "SampleRate" "Units" "Precision" "\n"], ","));

    for j = 1:nChannels
        [~, nsEntityInfo] = ns_GetEntityInfo(hFile, j);
        fprintf('Entity %d: Type %d\n', j, nsEntityInfo.EntityType)
        disp(nsEntityInfo)

        switch nsEntityInfo.EntityType
            case 0
                disp("Unknown entity!") 
            case 1
                [~, out] = ns_GetEventInfo(hFile, j);
            case 2
                [~, nsAnalogInfo] = ns_GetAnalogInfo(hFile, j);

                % disp(nsAnalogInfo)
                
                if contains(nsAnalogInfo.ProbeInfo, "Filter") && IGNORE_FILTER
                    continue
                end

                f_s = nsAnalogInfo.SampleRate;
                T_s = 1/f_s;
                n_samples = nsEntityInfo.ItemCount;
                n_intervals = ceil(n_samples / f_s / TIME_BLOCK_INTERVAL);
                n_samples_in_interval = round(TIME_BLOCK_INTERVAL / T_s);
                n_samples_last = mod(n_samples, n_samples_in_interval);

                fprintf(fileIDTxt, strjoin( ...
                    [j colCount nsEntityInfo.EntityLabel nsAnalogInfo.ProbeInfo f_s ...
                    nsAnalogInfo.Units SAVE_PRECISION "\n"], ","));
                colCount = colCount + 1;

                for k = 1:n_intervals
                    % Extract data from DDF files
                    if k == n_intervals
                        [~, ~, DDFdata] = ns_GetAnalogData( ...
                            hFile, j, (k-1) * n_samples_in_interval + 1, n_samples_last);
                    else
                        [~, ~, DDFdata] = ns_GetAnalogData( ...
                            hFile, j, (k-1) * n_samples_in_interval + 1, n_samples_in_interval);
                    end
                    fwrite(fileID, DDFdata, 'float32'); % Write data to BINary file
                    clear DDFdata;
                end

            case 3
                [~, out] = ns_GetSegmentInfo(hFile, j);
            case 4
                [~, out] = ns_GetNeuralInfo(hFile, j);
            otherwise
                disp("Invalid EntityType!")
        end
        
    end

    fclose(fileID); % Close the BINary file
    fclose(fileIDTxt); 

end


