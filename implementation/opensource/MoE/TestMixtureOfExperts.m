function [err, r] = TestMixtureOfExperts(problemType, vx, vr, v, m)
    if strcmpi(problemType, 'regression')
        ptype = 1;
    else
        ptype = 2;
    end
    outputCount = size(vr, 2);
    expertCount = size(m,1);
    sampleCount = size(vx, 1);
    vx = [ones(sampleCount, 1) vx];
    w = zeros(outputCount, expertCount, sampleCount);
    for j=1:outputCount
        w(j, :, :) = (v(:,:,j)*vx');
    end
    g = (exp(m*vx'));        
    g = g ./ repmat(sum(g, 1), expertCount, 1);
    y = zeros(sampleCount, outputCount);
    for i=1:sampleCount
        y(i, :) = ( w(:,:,i) * g(:,i));        
    end
    if ptype == 1
        err = sum((y - vr).^2) ./ sampleCount;
        r = y;
    else
        y = exp(y);
        cr = (y==repmat(max(y, [], 2), 1, outputCount));
        err = sum(sum(cr~=vr)) ./ (sampleCount .* outputCount);
        r = cr;
    end    
end