module SyntheticData

export rand_HMM_model, rand_HMM_data

using HMMTypes
using EmissionDistributions

using Distributions
using StatsBase
using Memoize


function rand_HMM_model(p :: Integer,
                        k :: Integer;
                        avg_range = .1,
                        sparsity = .3,
                        dwell_prob = 3/4)
    dists = Array(MvNormal, k)
    for i = 1:k
        mu = rand(p) * avg_range - avg_range/2;
        cov = rand_cov(p, sparsity)

        dists[i] = MvNormal(mu, cov)
    end

    states = map(dist -> HMMState(dist, true), dists)
    HMMStateModel(states, uniform_trans(k, dwell_prob))
end

function rand_HMM_data(n :: Integer,
                       p :: Integer,
                       k :: Integer,
                       model_generator = rand_HMM_model :: Function;
                       starting_distribution = Void,
                       sample = state_sample)
    model = model_generator(p, k)

    if starting_distribution == Void
        starting_distribution = vec(sum(model.trans, 2))
    end

    rand_HMM_data(n, p,
                  model;
                  starting_distribution = starting_distribution,
                  sample = state_sample)
end

function rand_HMM_data(n :: Integer,
                       p :: Integer,
                       model :: HMMStateModel;
                       starting_distribution = vec(sum(model.trans, 2)),
                       sample = state_sample)
    data = Array(Float64, p, n)

    labels = sample_label_series(n, model.trans, starting_distribution)

    for i = 1:n
        data[:, i] = sample(model.states[labels[i]]);
    end

    (data, labels, model)
end

function sample_label_series(n :: Integer,
                             trans :: Array{Float64, 2},
                             init :: Array{Float64, 1})
    init = init / sum(init)
    initial_state = rand(Categorical(init))
    sample_label_series(n, trans, initial_state)
end

function sample_label_series(n :: Integer,
                             trans :: Array{Float64, 2},
                             init :: Integer = 1)
    k = size(trans,1)
    dists = Array(Categorical, k)
    for i = 1:k
        dists[i] = Categorical(vec(trans[i, :]))
    end

    labels = Array(Int64, n)

    state = init
    for i = 1:n
        labels[i] = state;
        state = rand(dists[state]);
    end

    labels
end


function uniform_trans(k, prop)
    eye(k) * prop + ((1 - prop) / (k - 1)) * (ones(k, k) - eye(k));
end

function rand_cov(p)
    inv(cholfact(randInvcov(p)))
end

#found in GaussianMixtures, should produce a PSD matrix
function randInvcov(p)
    T = rand(p, p)
    inv(cholfact(T' * T / p))
end

function rand_cov(p, sparsity)
    # generate 30 psd matricies with aprox. correct sparsities
    aprox_invcovs = [aprox_sparse_psd_matrix(p, sparsity) for i = 1:30]

    # measure closeness to desired sparsity
    sparsity_errors = map(m -> abs(sparsity - mat_sparsity(m)), aprox_invcovs)

    # return inverse of closest
    closest_invcov = aprox_invcovs[indmin(sparsity_errors)]
    B = inv(cholfact(closest_invcov))
    normalize_determinant(B)
end

# TODO only measure sparsity of off-diags
function mat_sparsity (m; eps = 10e-6)
    p = size(m, 1)
    num_zero = length(filter(x -> x < eps, m + eye(p)))
    num_zero / (p*p-p)
end

function normalize_determinant(m :: Array{Float64, 2}, to = 1)
    c = (to/det(m))^(1/size(m, 1))
    c * m
end


function aprox_sparse_psd_matrix(p, sparsity)
    generator = () -> rand(p, p)
    p_sparsity = 1 - sqrt((1-sparsity) / p)
    P = sparsify_rand(generator, p_sparsity, X -> rank(X) == p)
    P' * P
end

function sparsify_rand(generator, sparify, valid)
    result = false

    while result == false
        M = generator()
        result = sparsify_mat(m, sparsity, valid)
    end

    result
end

# TODO only measure sparsity of off-diags
@memoize function sparsify_mat(m, sparsity, valid)
    p = size(m, 1)
    s = p * p - p;
    nzeros = s * sparsity;

    for i = 1:nzeros/2
        m = zero_rand_offdiag(m, valid);
        if (m == false)
            return false
        end
    end

    m
end

function sparsify_rand(generator, sparsity, valid)
    M = generator()

    s = size(M)[1] * size(M)[2];
    nzeros = s * sparsity;

    for i = 1:nzeros/2
        M = zero_rand_offdiag(M, valid);
        if(M == false)
            return sparsify_rand(generator, sparsity, valid);
        end
    end

    M
end

function zero_rand_offdiag(M, valid = isposdef)
    (n, m) = size(M);

    for i = shuffle(collect(1:n))
        for j = shuffle(collect(1:(i-1)))
            if (M[i, j] == 0)
                continue
            end

            temp = M[i, j]

            M[i, j] = 0.0;
            M[j, i] = 0.0;

            if(valid(M))
                return M
            else
                M[i, j] = temp
                M[j, i] = temp
            end
        end
    end

    return false
end

end
