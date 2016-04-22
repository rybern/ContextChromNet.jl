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

end
