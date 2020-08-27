function [ SS ] = yet_another_projection( S, k )
d = length(S);
i = 1;
while S(i) < 0
    S(i) = 0;
    i = i+1;
end
    SS = S;
SS(S>1)=1;
if sum(SS) <= k
    return
end
if S(end) <= 1
    i = 1;
    p_i = sum(S);
    while p_i > k
        i = i+1;
        SS = S-S(i-1);
        p_i = sum(SS(i:end));
    end
    if p_i == k
        SS(SS<0) = 0;
        return        
    end
    SS = S + (k- sum(S(i-1:end)))/(d-i+2);
    SS(SS>1) = 1; SS(SS<0) = 0;
    return
end
SS=S;
SS(S>1)=1; SS(S<0)=0;
if sum(SS)<=k
    return;
end
i = 1; j=1;
while 1+1e-6 - S(j+1) > 0
    j = j+1;
end
% jj = j;
while i <= d && j <=d-1
    s_ij = 1-S(j+1);
    while S(i) + s_ij <=0 && i<= j
        i = i+1;
    end
    if i>j
        j = j+1;
        continue;
    end
    SS = S + s_ij;
    p_ij = d-j+sum(SS(i:j));
    if p_ij > k
        j = j+1;
        continue;
    end
    if p_ij == k
        SS(SS>1) = 1; SS(SS<0) = 0;
        return
    end
%     if jj <= j-1
%         j = j-1; i= 1;
%     else
%         j = jj; i=1;
%     end
    i = 1;
    while S(i) + (k-(d-j)-sum(S(i:j)))/(j-i+1) <= 0
        i = i+1;
    end
    SS = S + (k-(d-j)-sum(S(i:j)))/(j-i+1);
    SS(SS>1) = 1; SS(SS<0) = 0;
    return
end
while S(i) + (k-(d-j)-sum(S(i:j)))/(j-i+1) <= 0
    i = i+1;
end
SS = S + (k-(d-j)-sum(S(i:j)))/(j-i+1);
SS(SS>1) = 1; SS(SS<0) = 0;
end

