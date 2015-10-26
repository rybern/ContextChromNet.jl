module biogrid_eval

export eval_partition, biogrid_score_matrix, eval_promoter, parse_biogrid_score, pr_curve, auc, pr_auc, read_biogrid_score, weighted_max, print_something

using splitting
using analysis
using loading

# data: sample x track matrix
# partition: boolean matrix
function eval_promoter ()
    header = load_header(default_header)

    full_invcor = read_float_arr("data/inverse_data_correlation.Float64", 1415, 1415)
    a_invcor = read_float_arr("data/inverse_promoter_correlation.Float64", 1415, 1415);
    b_invcor = read_float_arr("data/inverse_nonpromoter_correlation.Float64", 1415, 1415);
    max_invcor = weighted_max(a_invcor, b_invcor, 1);

    full_score = biogrid_score_matrix(full_invcor, header);
    a_score = biogrid_score_matrix(a_invcor, header);
    b_score = biogrid_score_matrix(b_invcor, header);
    max_score = biogrid_score_matrix(max_invcor, header);

    a_score, b_score, max_score, full_score
end

# data: sample x track matrix
# partition: boolean matrix
function eval_partition (partition, data, header)
    print("Splitting data...")
    a, b = split_data(partition, data);
    print("done.\n")

    print("Calculating covariances...")
    a_cor = bcor(a);
    a_invcor = inv(a_cor);
    b_cor = bcor(b);
    b_invcor = inv(b_cor);
    max_invcor = weighted_max(a_invcor, b_invcor, 1);
    print("done.\n")

    max_score = biogrid_score_matrix(max_invcor, header);
    a_score = biogrid_score_matrix(a_invcor, header);
    b_score = biogrid_score_matrix(b_invcor, header);
    
    a_score, b_score, max_score
end

function weighted_max (v1, v2, w)
    r, c = size(v1)
    v_max = zeros(r, c)
    for y = 1:r
        for x = 1:c
            if abs(v1[y,x]) * w > abs(v2[y,x])
                v_max[y,x] = v1[y,x]
            else 
                v_max[y,x] = v2[y,x]
            end
        end
    end
    v_max
end

function full_weighted_max (v1, v2, w)
    weighted_max(v1*w, v2, 1)
end



function precision_recall (true_pos, false_pos, false_neg)
    p = true_pos / (true_pos + false_pos);
    r = true_pos / (true_pos + false_neg);
    p, r
end

# Scott's GitHub has his version to compare
function pr_curve (truth)
    n = size(truth)[1];
    
    true_pos = 0;
    false_pos = 0;
    false_neg = count(identity, truth);

    precision = zeros(n);
    recall = zeros(n);

    for i = 1:n
        if truth[i]
            true_pos = true_pos + 1;
            false_neg = false_neg - 1;
        else
            false_pos = false_pos + 1;
        end 

        precision[i], recall[i] = precision_recall(true_pos, false_pos, false_neg);
    end

    precision, recall
end

function auc (x, y)
    sum = 0;
    for i = 1:(size(x)[1]-1)
        y_avg = 0.5*(y[i] + y[i+1]);
        rectangle = (x[i + 1] - x[i]) * y_avg
        sum = sum + rectangle;
    end
    sum
end

function pr_auc (truth)
    print("pr auc...")
    p, r = pr_curve(truth);
    res = auc(r, p);
    print("done\n")
    res
end

function eval_partition (partition)
    data = load_data(default_data);
    header = load_header(default_header);
    eval_partition (partition, data, header);
end

function with_temp(fn)
    tempname, tempfile = mktemp();
    fn(tempfile)
    close(tempfile)
    tempname
end

function print_something()
    print("hi, two, three")
end

function biogrid_score_matrix(matrix, header)
    temp_score_file = "temp_score_file";

    groupgm_file = matrix_csv(matrix, header)
    biogrid_score_file(groupgm_file, temp_score_file)
    truth = read_biogrid_score(temp_score_file);
    pr_auc(truth)
end

function biogrid_score_file(groupgm_file, output_file)
    print("scoring now...")
    score_string = run(`/home/ryan/Documents/CompBio/test_csv_file.sh $groupgm_file $output_file`)
    print("done\n")
end

function matrix_csv(matrix, header, file)
    print("matrix_csv\n")

    # write header
    write(file, join(header, ","))
    write(file, "\n")

    # write matrix
    for r = 1:size(matrix)[1]
        write(file, join(matrix[r,:], ","))
        write(file, "\n")
    end
end

function matrix_csv(matrix, header)
    with_temp(file -> matrix_csv(matrix, header, file))
end

function parse_biogrid_score (output)
    readdlm(IOBuffer(output), ',')
end

function read_biogrid_score(output_file)
    print("reading results...")
    data = readall(output_file);
    print("done\n")
    print("parsing results...")
    results = parse_biogrid_score(data);
    results[:, 1] = map(x->abs(float64(x)), results[:, 1]);
    print("done\n")
    print("sorting...")
    results = sortrows(results, rev=true)
    print("done\n")
    map(x -> x == 1, results[:,2])
end

end
