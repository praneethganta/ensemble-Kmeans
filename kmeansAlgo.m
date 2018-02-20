%add no. of cluster values to the below vector if required
cluster = [3 5 7];
for j = cluster(1,:)
    kmeansData = importdata('seeds.txt');
    clusterPoints = randperm(size(kmeansData,1),j);
    clusterValues = kmeansData(clusterPoints,:);
    prevError = 0;
    squareError = 0;
    squareErrorDiff = 0;
    count = 0;

    while count < 100
        distanceMatrix = pdist2(clusterValues,kmeansData,'euclidean');
        [val,row] = min(distanceMatrix,[],1);
        row = transpose(row);

        for i = 1:j
            clusterIndex = find(row == i);
            clusterValues(i,:) = mean(kmeansData(clusterIndex,:));
        end
        val = val.^2;
        squareError = sum(val);
        if count ~= 0
            squareErrorDiff = prevError - squareError;
             if squareErrorDiff < 0.001
                break;
             end
        end
        prevError = squareError;
        count = count + 1;
    end
    fprintf('Sum of Squared mean error when k =%d is %f \n',j,squareError);
end