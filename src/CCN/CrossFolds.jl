module CrossFolds

export partition_array, partition_interval, fold_interval

using ArrayViews

function partition_interval(n, k)
    partition = Array(UnitRange{Int64}, k)
    base_size = div(n, k)
    extras = n % k

    agg = 1

    for i = 1 : k
        agg_before = agg
        agg += base_size + (i > extras ? 0 : 1)
        partition[i] = agg_before:agg - 1
    end

    partition
end

function fold_indices(k)
    not = i -> j -> i != j
    interval = collect(1:k)
    [(filter(not(i), interval), i) for i = interval]
end

function fold_interval(n, k)
    partition = partition_interval(n, k)
    fold = fold_indices(k)

    [(partition[fold[i][1]], partition[fold[i][2]]) for i = 1:k]
end



function partition_array(arr, k; dim = 1)
    n = size(arr, dim)
    partition = partition_interval(n, k)
    
    if(dim == 1)
        return [view(arr, partition[i]) for i = 1:k]
    elseif(dim == 2)
        return [view(arr, :, partition[i]) for i = 1:k]
    end
end

end
