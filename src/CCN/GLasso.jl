module GLasso
export glasso_cov, glasso, universal_lambda

using BaumWelchUtils

using Rif
using Distributions
using StatsBase
using LambdaOptimization

using GraphLasso

function glasso(cov, alpha)
    tol = 1e-4
    graphlasso(cov, alpha, penalize_diag = false, maxit = 1000, tol = tol)[1]
end

#new
function glasso_cov(data, weights, lambda = universal_lambda(size(data)...))
    n = size(data, 2)
    fullCov = cov(data, WeightVec(weights), vardim=2);

    sumW = sum(weights)
    piK = sumW / n;
    rho = 2 * sqrt(piK) * lambda / sumW

    glasso(fullCov, rho)
end

function universal_lambda(p, n)
    sqrt(2 * n * log(p)) / 2
end

#old
function glasso_cov_old(data, weights, lambda = universal_lambda(size(data)...))
    n = size(data, 2)
    fullCov = cov(data, WeightVec(weights), vardim=2);

    sumW = sum(weights)
    piK = sumW / n;
    rho = 2 * sqrt(piK) * lambda / sumW

    glasso_old(fullCov, rho)
end

function glasso_old(cov, rho)
    if (rho == 0)
        cov
    else
        wi = false
        err = false
        try
            for i = 1:4
                try
                    wi = rifGlasso(cov, rho);

                    break
                catch e
                    continue
                end
            end
        catch e
            error(e)
        end
        wi = force_pos_def(wi)

        w = inv(cholfact(wi))

        if(false && !isposdef(w))
            println("inverse of glasso result is not pos def!")
            println("inv of glasso result:")
            println(w)
            println("glasso result:")
            println(wi)
            println("orginal cov:")
            println(cov)
            error("not pos def")
        end

        w
    end
end

function rifGlasso(cov, rho)
    #        Rif.initr()
    # Initialize R environment
    glassor = Rif.importr("glasso")

    # Get reference to R environment
    ge = Rif.getGlobalEnv()

    # Load variables into R environment
    covr = Rif.RArray{Float64, 2}(cov);
    ge["covr"] = covr
    Rif.R("rho <- $rho")

    # Run GLasso
    Rif.R("res <- glasso(covr, rho, penalize.diagonal = FALSE)")

    # Extract R result and copy results into Julia array
    wir = Rif.R("res[[\"wi\"]]")
    wi = Array(Float64, size(cov))
    for i = 1:size(cov)[1]
        for j = 1:size(cov)[2]
            wi[i, j] = wir[i, j]
        end
    end
    #(wi + wi')/2

    wi
end

end



