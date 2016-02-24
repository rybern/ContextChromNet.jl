# fit_emissions :: data -> gamma -> State
# emission_pdf :: data -> gamma -> log probability

module EmissionDistributions
export fit_full_cov, fit_diag_cov, fit_glasso, dist_log_pdf, state_sample, fit_glasso_old

using HMMTypes
using GLasso
using BaumWelchUtils

using Distributions
using StatsBase

function state_sample(state)
    if state.active
        rand(state.dist)
    else
        0
    end
end

function log_pdf_dist_to_state(log_pdf_dist)
    function log_pdf_state(r, state, data)
        if state.active
            log_pdf_dist(r, state.dist, data)
        else
            fill!(r, 0)
        end
    end
end

function pmap_(f, lst1, lst2, pslist)
    np = nprocs()  # determine the number of processes available
    n = length(lst1)
    results = cell(n)
    i = 1
    # function to produce the next work item from the queue.
    # in this case it's just an index.
    nextidx() = (idx=i; i+=1; idx)
    @sync begin
        for p=pslist
            if p != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx > n
                            break
                        end
                        results[idx] = remotecall_fetch(p, f, lst1[idx], lst2[idx])
                    end
                end
            end
        end
    end
    results
end

num_fit_procs = 3

function fit_dist_to_state(fit_dist)
    function fit_state(data,
                        weights,
                        old_state)
#    function fit_state {N <: Number} (data :: AbstractArray{N, 2},
#                                      weights :: Array{Float64, 1},
#                                      old_state :: HMMState)
#        println(sum(data))
#        println(sum(weights))
#        println(typeof(old_state))
#        flush(STDOUT)
        weights_sum = sum(weights)
        weights_mean = weights_sum / length(weights)
        if weights_sum < 1 || weights_mean < 1e-20 || old_state.active == false
            HMMState(Void, false)
        else
            HMMState(fit_dist(data, weights), true)
        end
    end

    function fit_states{N <: Number}(data :: AbstractArray{N, 2},
                                       gamma :: Array{Float64, 2},
                                       old_states :: Array{HMMState, 1} =
                                       [HMMState(Void, true)
                                        for i = 1:size(gamma, 1)])
        k = length(old_states)

        gamma = gamma'

        # should be view, haven't looked into how to get it to work
        # with cov
        weights = [gamma[:, state_ix] for state_ix = 1:k]

        new_states = pmap_((weight, state) -> fit_state(data, weight, state),
                           weights, old_states,
                           procs()[1:min(nprocs(), num_fit_procs)])

        convert(Array{HMMState, 1},
                new_states)

#        new_states = Array(HMMState, k)

#        for state_ix = 1:k
#            weights = gamma[:, state_ix]
#            new_states[state_ix] = fit_state (data, weights, old_states[state_ix])
#        end

#        new_states
    end
end

function fit_dist_full_cov{N <: Number}(data :: AbstractArray{N, 2},
                                        weights)
    (mu, cov) = mean_and_cov(data, WeightVec(weights), vardim=2)
    safe_mv_normal(mu, cov)
end

function fit_dist_diag_cov{N <: Number}(data :: AbstractArray{N, 2},
                                        weights)
    p = size(data, 1)

    (mu, cov) = mean_and_cov(data, WeightVec(weights), vardim=2)
    safe_mv_normal(mu, cov .* eye(p))
end

function fit_dist_glasso{N <: Number}(data :: AbstractArray{N, 2},
                                      weights)
    mu, cov_ = mean_and_cov(data, WeightVec(weights), vardim=2)
    cov = glasso_cov(data, weights)

    safe_mv_normal(mu, cov)
end

function fit_dist_glasso_old{N <: Number}(data :: AbstractArray{N, 2},
                                      weights)
    mu, cov_ = mean_and_cov(data, WeightVec(weights), vardim=2)
    cov = GLasso.glasso_cov_old(data, weights)

    safe_mv_normal(mu, cov)
end

fit_full_cov = fit_dist_to_state(fit_dist_full_cov)
fit_diag_cov = fit_dist_to_state(fit_dist_diag_cov)
fit_glasso = fit_dist_to_state(fit_dist_glasso)
fit_glasso_old = fit_dist_to_state(fit_dist_glasso_old)
dist_log_pdf = log_pdf_dist_to_state(logpdf!)

function safe_mv_normal(mu :: Array{Float64},
                        cov :: Array{Float64, 2},
                        check_singular = false)
    try
        if (check_singular && det(10000*cov) == 0)
            println("Singular matrix encountered. Sample not long enough.")
            println("Temporarily using identity cov.")
            cov = eye(size(cov,1))
        end

        MvNormal(vec(mu), cov)
    catch e
        cov_ = force_pos_def(cov)
        try
            MvNormal(vec(mu), cov_)
        catch e2
            error("repeat offender")
        end
    end
end

end
