module RealValidation

export learn_CCN

using Loading
using EmissionDistributions
using BaumWelch
using SimpleLogging


estimate_filename = "estimate.dump"
loglikelihood_filename = "loglikelihood.dump"
model_filename = "model.dump"

function dump_results_base(base_directory)
    if !isdir(base_directory)
        mkdir(base_directory)
    end

    function dump_results(dir_name, estimate, model, loglikelihood)
        output_dir = join([base_directory, dir_name], "/")

        if !isdir(output_dir)
            mkdir(output_dir)
        end

        estimate_fp = join([output_dir, estimate_filename], "/")
        loglikelihood_fp = join([output_dir, loglikelihood_filename], "/")
        model_fp = join([output_dir, model_filename], "/")

        open(s -> serialize(s, estimate), estimate_fp, "w")
        open(s -> serialize(s, loglikelihood), loglikelihood_fp, "w")
        open(s -> serialize(s, model), model_fp, "w")
    end
end

function learn_CCN(data,
                   k,
                   output_dir = Void;
                   num_runs = 5,
                   fit_emissions = fit_full_cov,
                   #(data, k) -> (estimate, model, log-likelihood)
                   model_optimizer = (data, k) -> baum_welch(num_runs,
                                                             data,
                                                             k,
                                                             fit_emissions,
                                                             verbose = true,
                                                             result_writer = output_dir == Void ? Void : dump_results_base(output_dir)),
                    verbose = true)
    if verbose
        logstrln("Starting real validation")
    end

    model_optimizer(data, k)
end

function prune_output_dir(dir)
    function is_output(output_dir)
        full_dir = join([dir, output_dir], "/")

        if !isdir(full_dir)
            return false
        end

        subdirs = readdir(full_dir)
        return (in(estimate_filename, subdirs) &&
                in(loglikelihood_filename, subdirs) &&
                in(model_filename, subdirs))
    end

    function output_ll(output_dir)
        open(deserialize, join([dir, output_dir, loglikelihood_filename], "/"))
    end

    subfiles = readdir(dir)
    outputs = filter(is_output, subfiles)
    lls = map(output_ll, outputs)
    perm = sortperm(-lls)
    sorted_outputs = outputs[perm]

    good_dir = sorted_outputs[1]
    for file in readdir(join([dir, good_dir], "/"))
        mv(join([dir, good_dir, file], "/"),
           join([dir, file], "/"),
           remove_destination = true)
    end

    for output_dir in outputs
        rm(join([dir, output_dir], "/"), recursive = true)
    end

    open(s -> serialize(s, lls), join([dir, "all_lls.dump"], "/"), "w")
end

end
