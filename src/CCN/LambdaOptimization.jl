module LambdaOptimization

export cup_direct_search, disjoint_interval_mean_and_cov, disjoint_interval_mean, build_lambda_likelihood, adaptive_glasso_mixture

using StatsBase
using CrossFolds
using Distributions

# basically binary search
# lower_bound should be the minimum value likely
# step should be some step in the right direction
function cup_direct_search(f, step, lower_bound, max_step_changes, max_strides)
    x0 = lower_bound
    y0 = f(x0)
    iter = 0

    for stide = 1:max_strides
        x1 = x0 + step
    
        iter += 1
        if ((y1 = f(x1)) > y0)
            x0 = x1
            y0 = y1
        else
            break
        end
    end

    for shorten = 1:max_step_changes
        step /= 2

        x_right = x0 + step
        x_left = x0 - step
        
        iter += 2
        if(x_right > lower_bound && (y_right = f(x_right)) > y0)
            x0 = x_right
            y0 = y_right
            iter -= 1
        elseif(x_left > lower_bound && (y_left = f(x_left)) > y0)
            x0 = x_left
            y0 = y_left
        end
            #println ("didn't go right, ($x_right, $y_right) < ($x0, $y0)") 
    end

    println("selected parameter $x0 with score $y0 after $iter iterations")
    (x0, y0)
end

function rho(lambda, sum_w, n)
    2 * sqrt(sum_w/n) * lambda / sum_w
end

#can take out some repeated weight vector computations, like sums, props
function build_lambda_likelihood(data, weights, k_folds)
    (p, n) = size(data)
    k = size(weights, 2)

    folds = fold_interval(n, k_folds)
    
    weight_sums = [sum(weights[:, i]) for i = 1:k]
    weight_props = weight_sums / sum(weight_sums)

    test_sets = [[view(data, :, folds[fold][2]) .* repeat(weights[folds[fold][2], i], inner = [1, p])' for fold = 1:k_folds] for i = 1:k]
    test_weights = [[sum(view(weights, folds[fold][2], i)) for fold = 1:k_folds] for i = 1:k]

    train_moments = [[disjoint_interval_mean_and_cov(data, weights[:, i], folds[fold][1]) for fold = 1:k_folds] for i = 1:k]    
    train_weights = [weight_sums[i] - test_weights[i] for i = 1:k]
    train_weight_props = [train_weights[i] / weight_sums[i] for i = 1:k]

    function lambda_likelihood(lambda)
        @parallel (x, y) -> (x + y) for (i, fold) = [(i, fold) for i = 1:k, fold = 1:k_folds]

            coeficient = train_weight_props[i][fold] * weight_props[i]

            (mu, cov) = train_moments[i][fold]
            r = rho(lambda, train_weights[i][fold], n)

            fit_cov = glasso(cov, r)

            model = safeMvNormal(mu, fit_cov)
            likelihood = loglikelihood(model, test_sets[i][fold])
            likelihood * coeficient
            
        end
    end
end

function disjoint_interval_mean(arr, weights, intervals)
    weight_sums = [sum(view(weights, interval)) for interval = intervals]/sum(weights)
    means = [vec(mean(view(arr, :, intervals[i]), WeightVec(weights[intervals[i]]), 2)) * weight_sums[i] for i = 1:size(intervals, 1)]

    sum(means, 1)[1]#, size(intervals, 1), 1)
end

function disjoint_interval_mean_and_cov(arr, weights, intervals)
    mu = disjoint_interval_mean(arr, weights, intervals)

    covs = [cov(view(arr, :, interval), WeightVec(weights[interval]), vardim = 2, mean = mu)
            for interval = intervals]

    weight_sums = [sum(view(weights, interval)) for interval = intervals]
    weight_props = weight_sums / sum(weight_sums)
    weighted_avg_covs = sum(weight_props .* covs)
    (mu, weighted_avg_covs)
end

function adaptive_glasso_mixture(data, weights)
    println("building ll penalty...")
    ll = build_lambda_likelihood(data, weights, 5)
    println("doing optimization...")
    lambda = cup_direct_search(ll, lambdaUni(size(data)...), 0, 30, 30)[1]
    println("found lambda $lambda, universal is ", lambdaUni(size(data)...))
    glasso_mixture(data, weights, lambda)
end

function glasso_mixture(data, weights, lambda)
    n = size(data, 2)
    k = size(weights, 2)
    moments = [mean_and_cov(data, WeightVec(weights[:, i]), vardim = 2) for i = 1:k]
    weight_sums = [sum(weights[:, i]) for i = 1:k]

    rhos = [rho(lambda, weight_sums[i], n) for i = 1:k]
    models = [safeMvNormal(moments[i][1], glasso(moments[i][2], rhos[i])) for i = 1:k]
end

end
