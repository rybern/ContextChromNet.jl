module ExtendedLogs

export logzero, eexp, eln, elnsum, elnsum, eln_arr, eexp_arr, eln, elnsum_arr, elnsum_arr#, parallel_emissions

#using ArrayViews

logzero = NaN

function eexp(lnx)
    if lnx != lnx
        0
    else 
        exp(lnx)
    end
end

function eln(lnx)
    if lnx == 0
        logzero
    else
        log(lnx)
    end
end

function eln_arr(x)
    map(eln, x)
end

function eexp_arr(x)
    map(eexp, x)
end


function elnsum(a, b)
    if a != a
        b
    elseif b != b
        a
    elseif a > b
        a + log(1 + exp(b - a))
    else 
        b + log(1 + exp(a - b))
    end
end

# probably very inefficient
function elnsum_arr(v)
    reduce(elnsum, v)
end

function elnprod_arr(v)
    sum(v)
end
   
function a()
    println(1, 2, 3)
    4 + 5
end

@everywhere function a()
    println(1, 2, 3)
    4 + 5
end


end
