function res = get_data(filename,key)
    fileID = fopen(filename,'r');
    C= textscan(fileID,'%s',17,'Delimiter',',');
    fclose(fileID);
    for i=1:numel(C{1})
        if strcmp(C{1}(i),key)
            idx = i;
            break
        end
    end
    data = csvread(filename,1,0);
    res = data(:,idx);
end
