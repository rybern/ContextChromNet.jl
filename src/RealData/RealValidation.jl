module RealValidation

export learn_CCN

using Loading
using EmissionDistributions
using BaumWelch
using Logging


estimate_filename = "estimate.dump"
loglikelihood_filename = "loglikelihood.dump"
model_filename = "model.dump"

function learn_CCN (data,
                    k,
                    output_dir = Nothing;
                    num_runs = 5,
                    fit_emissions = fit_full_cov,
                    #(data, k) -> (estimate, model, log-likelihood)
                    model_optimizer = (data, k) -> baum_welch(num_runs,
                                                              data,
                                                              k,
                                                              fit_emissions,
                                                              verbose = true))
    # make sure output dir is available
    if output_dir != Nothing && !isdir(output_dir)
        mkdir(output_dir)
    end

    if verbose
        logstrln("Starting real validation")
    end

    (estimate, model, ll) = model_optimizer(data, k)

    if output_dir != Nothing
        estimate_filepath = "$output_dir/$estimate_filename"
        loglikelihood_filepath = "$output_dir/$loglikelihood_filename"
        model_filepath = "$output_dir/$model_filename"
    end

    (estimate, model, ll)
end

end 