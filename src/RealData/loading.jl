module loading

export loadFilteredData, loadFilteredHeader, loadData, loadHeader, loadMetadata, loadMapping, loadPairs, withData, withHeader, withDataHeader, loadFiltered

using JSON

function withData(f,
                  n = 10000;
                  dataPredicate = x -> x["cellType"] == "K562", 
                  averagedTargets = false)
    (data, header) = loadFilteredData(filterBy = dataPredicate, firstN = n)
    f(data)
end

function withHeader(f,
                    n = 10000;
                    dataPredicate = x -> x["cellType"] == "K562", 
                    averagedTargets = false)
    header = loadFilteredHeader(filterBy = dataPredicate, firstN = n)

    f(header)
end

function withDataHeader(f,
                        n = 10000;
                        dataPredicate = x -> x["cellType"] == "K562",
                        averagedTargets = false)
    (data, header) = loadFiltered(filterBy = dataPredicate, firstN = n)
    f(data, header)
end

function filterIndices(filterBy, header, metadata)
    allowedKeys = Set(filterKeysByValue(metadata, filterBy)); 

    filter(i -> in(header[i], allowedKeys), [1:size(header)[1]]);
end

function loadFiltered(; filterBy = x -> x["cellType"] == "K562",
                      dir_name = "ChromNet_build1/ENCODE_build1.ChromNet/",
                      firstN = 0,
                      averagedTargets = false)
    header_all = loadHeader("$dir_name/header")
    metadata = loadMetadata("$dir_name/metadata")

    allowedIndices = filterIndices(filterBy, header_all, metadata)

    data_all = loadData("$dir_name/matrix")

    data = firstN == 0 ? data_all[allowedIndices, :] : data_all[allowedIndices, 1:firstN]
    data = baToBool(data)

    header = header_all[allowedIndices]

    averagedTargets ? averageTargets(data, header) : (data, header)
end

function averageTargets(data, experiments, mapping = loadMapping())
    mappableExperiments = filter(e -> in(e, keys(mapping)), experiments)
    mappableExpSet = Set(mappableExperiments)

    proteins = unique(map(exp -> mapping[exp], mappableExperiments))
    
    indexMapping = Dict{String, Array{Int64, 1}}()
    for p = proteins
        indexMapping[p] = []
    end

    for ei = 1:size(experiments, 1)
        e = experiments[ei]
        if(!in(e, mappableExpSet))
            continue
        end
        protein = mapping[e]
        push!(indexMapping[protein], ei)
    end

    pdata = Array(Float64, size(proteins, 1), size(data, 2))

    for pi = 1:size(proteins, 1)
        experimentIds = indexMapping[proteins[pi]]
        rows = [data[ei, :] for ei = experimentIds]
        avg_row = mean(vcat(rows...), 1)
        pdata[pi, :] = avg_row
    end
        
    pdata, proteins
end

function loadFilteredHeader(; filterBy = x -> x["cellType"] == "K562",
                            dir_name = "ChromNet_build1/ENCODE_build1.ChromNet/",
                            firstN = 0)
    header = loadHeader("$dir_name/header")
    metadata = loadMetadata("$dir_name/metadata")

    allowedIndices = filterIndices(filterBy, header, metadata)

    header[allowedIndices]
end



function filterKeysByValue(mapping, filterBy)
    filter(k -> filterBy(mapping[k]), keys(mapping))
end

function loadHeader(sourceName = "ChromNet_build1/ENCODE_build1.ChromNet/header")
    vec(readdlm("$sourceName"))
end

function loadData(sourceName = "ChromNet_build1/ENCODE_build1.ChromNet/matrix")
    f = open("$sourceName")
    N = read(f, Int64)
    P = read(f, Int64)
    data = BitArray(N, P) # uninitialized

    # julia arrays are column major
    read!(f, data)
    close(f)

    data
end

function baToBool(m)
    convert(Array{Bool}, m)
end

function baToFloat(m)
    (n, p) = size(m)
    [float(m[i, j]) for i = 1:n, j = 1:p]
end
    

function loadMetadata(sourceName = "ChromNet_build1/ENCODE_build1.ChromNet/metadata")
    f = open(sourceName)
    m = JSON.parse(f)
    close(f)
    m
end

# experiment -> target (for reverse = false)
function loadMapping(sourceName = "uniprot.mapping";
                      reverse = false)
    mappingMatrix = open(readcsv, sourceName)
    if reverse
        [mappingMatrix[i, 2] => mappingMatrix[i, 1] for i = 1:size(mappingMatrix)[1]]    
    else
        [mappingMatrix[i, 1] => mappingMatrix[i, 2] for i = 1:size(mappingMatrix)[1]]
    end
end

function loadPairs(sourceName = "biogrid_human_swissprot.csv")
    pairsMatrix = open(readcsv, sourceName)
    pairs = collect(zip(pairsMatrix[:, 1], pairsMatrix[:, 2]));
    reverse = map(p -> (p[2], p[1]), pairs)
    Set([pairs, reverse])
end

function loadPairs2(pairsSource = "biogrid_human_swissprot.csv",
                     mapping = "uniprot.mapping";
                     nameWhitelist = [])
    pairsMatrix = open(readcsv, pairsSource)
    nameMap = loadMapping(mapping, reverse = true)

    nameSet = Set(collect(keys(nameMap)))

    useWhitelist = false
    whitelist = Set(nameWhitelist)
    if (nameWhitelist != [])
        useWhitelist = true
    end

    function translatePair(a, b)
        if (in(a, nameSet) && in(b, nameSet))
            a_ = nameMap[a];
            b_ = nameMap[b];
            if (a_ != b_ && (!useWhitelist || (in(a_, whitelist) && in(b_, whitelist))))
                [(a_, b_), (b_, a_)]
            else
                []
            end
        else 
            []
        end
    end

    pairs = zip(pairsMatrix[:, 1], pairsMatrix[:, 2]);
    translatedPairs = [translatePair(pair...) for pair = pairs]
    flatPairs = reduce(vcat, translatedPairs)

    pairSet = Set(flatPairs)
end

function saveData(file_name, data)
    f = open("$file_name", "w")
    N = size(data, 1)
    P = size(data, 2)
    write(f, N)
    write(f, P)

    # julia arrays are column major
    write(f, data)
    close(f)
end

function loadBinned(sourceName)
    f = open(sourceName)
    arr = Float64[]
    while !eof(f)
        a=readline(f)
        push!(arr,parsefloat(Float64, a))
    end
    close(f)
    arr
end

function saveBinned(file_name, data)
    f = open(file_name, "w");
    for d = data
        write(f, string(d, "\n"));
    end
    close(f);
end


function loadSplitTrack(track_file)
    map(x -> x > 0, load_binned(track_file));
end


# Defaults

base_dir = "/home/ryan/Documents/CompBio/";

default_promoter_track = "$base_dir/promoterStanford.binned"
default_data = "$base_dir/data/binary.BitArray/matrix"
default_header = "$base_dir/data/binary.BitArray/header"
default_promoter = "$base_dir/data/promoter.BitArray"
default_nonpromoter = "$base_dir/data/nonpromoter.BitArray"
subsampled_promoter = "$base_dir/data/subsampled_promoter.BitArray"

function loadMatricies()
    header = loadHeader(default_header)
    promoter = loadData(default_promoter)
    nonpromoter = loadData(default_nonpromoter)
    promoter, nonpromoter, header
end

end
