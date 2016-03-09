module SynthResults

function display_ticks(vars, ticks)
    for (vari, var) = enumerate(vars)
        println("$vari | $var:")
        for (ticki, tick) = enumerate(ticks[vari])
            println("\t$ticki | $tick")
        end
    end
end

# Typical output:
#
#
# model_optimizer (1):
# 	Full Covariance BW (1)
# 	True model (2)
# data_generator (2):
# 	Generating density 1.0 (1)
# 	Generating density 0.5 (2)
# 	Generating density 0.1 (3)
# validation_measure (3):
# 	Label Accuracy (1)
# 	Edge Accuracy (2)
# 	Test Log-Likelihood (3)
# 	Network Enrichment (4)
# iteration (4):
# 	1 (1)
# 	2 (2)
# 	3 (3)
# 	4 (4)
# 	5 (5)

function merge_models_files(result_file1, result_file2, output_file)
    results1 = open(deserialize, result_file1)
    results2 = open(deserialize, result_file2)
    results = merge_models(results1, results2)
    open(s -> serialize(s, results), output_file, "w")
end

function merge_models(results1, results2)
    (vars1, ticks1, res1, specs1) = results1
    (vars2, ticks2, res2, specs2) = results2

    @assert vars1 == vars2
    @assert vars1 == ("model_optimizer", "data_generator", "validation_measure", "iteration")

    variable_lengths = map(length, ticks1)
    variable_lengths[1] += length(ticks2[1])

    result_tensor = Array(Any, variable_lengths...)

    for a = 1:length(ticks1[1])
        for b = 1:length(ticks1[2])
            for c = 1:length(ticks1[3])
                for d = 1:length(ticks1[4])
                    result_tensor[a,b,c,d] = res1[a,b,c,d]
                end
            end
        end
    end

    for a = 1:length(ticks2[1])
        for b = 1:length(ticks2[2])
            for c = 1:length(ticks2[3])
                for d = 1:length(ticks2[4])
                    result_tensor[length(ticks1[1]) + a, b, c, d] = res2[a,b,c,d]
                end
            end
        end
    end

    ticks = ticks1
    ticks[1] = [ticks1[1]; ticks2[1]]

    results = (vars1, ticks, result_tensor, specs1);
end

function permute_models_files(result_file, output_file, permutation)
    results = open(deserialize, result_file)
    results = permute_models(results, permutation)
    open(s -> serialize(s, results), output_file, "w")
end

function permute_models(results, permutation)
    (vars, ticks, res, specs) = results

    @assert vars == ("model_optimizer", "data_generator", "validation_measure", "iteration")

    new_ticks = copy(ticks)
    permute!(new_ticks[1], permutation)

    res_new = Array(Any, size(res)...)
    for (new_ix, old_ix) = enumerate(permutation)
        res_new[new_ix, :, :, :] = res[old_ix, :, :, :]
    end

    (vars, new_ticks, res_new, specs)
end

end
