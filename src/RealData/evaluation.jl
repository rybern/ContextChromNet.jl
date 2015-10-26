module evaluation

export prCurve

function prCurve (truth)
    n = size(truth)[1];
    
    true_pos = 0;
    false_pos = 0;
    false_neg = count(identity, truth);

    precision = zeros(n);
    recall = zeros(n);

    for i = 1:n
        if truth[i]
            true_pos += 1
            false_neg -= 1
        else
            false_pos += 1
        end 

        precision[i], recall[i] = precision_recall(true_pos, false_pos, false_neg);
    end

    precision, recall
end

function precision_recall (true_pos, false_pos, false_neg)
    p = true_pos / (true_pos + false_pos);
    r = true_pos / (true_pos + false_neg);
    p, r
end

end
