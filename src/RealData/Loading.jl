module Loading

export with_data, with_data, with_data_header, load_filtered, load_data, load_header, load_pairs, load_metadata, load_mapping

using JSON

default_basedir = "data/ChromNet/ENCODE_build1.ChromNet"

function with_data(f :: Function,
                   n = Nothing;
                   dataPredicate = x -> x["cellType"] == "K562")
    (data, header) = load_filtered(filterBy = dataPredicate,
                                   firstN = n)

    f(data)
end

function with_header(f,
                     n = Nothing;
                     dataPredicate = x -> x["cellType"] == "K562")
    header = load_filtered_header(filterBy = dataPredicate,
                                  firstN = n)

    f(header)
end

function with_data_header(f,
                          n = Nothing;
                          dataPredicate = x -> x["cellType"] == "K562")
    (data, header) = load_filtered(filterBy = dataPredicate,
                                   firstN = n)

    f(data, header)
end

function filter_indices(filterBy, header, metadata)
    allowedKeys = Set(filter_keys_by_value(metadata, filterBy)); 

    filter(i -> in(header[i], allowedKeys), [1:size(header)[1]]);
end

function load_filtered(; filterBy = x -> x["cellType"] == "K562",
                       dir_name = default_basedir,
                       firstN = Nothing)
    header_all = load_header("$dir_name/header")
    metadata = load_metadata("$dir_name/metadata")

    allowedIndices = filter_indices(filterBy, header_all, metadata)

    data_all = load_data("$dir_name/matrix")

    data = firstN == Nothing ? data_all[allowedIndices, :] : data_all[allowedIndices, 1:firstN]
    data = convert(Array{Bool}, data)

    header = header_all[allowedIndices]

    (data, header)
end

function load_filtered_header(filterBy = x -> x["cellType"] == "K562";
                              dir_name = default_basedir,
                              firstN = Nothing)
    header = load_header("$dir_name/header")
    metadata = load_metadata("$dir_name/metadata")

    allowedIndices = filter_indices(filterBy, header, metadata)

    header[allowedIndices]
end

function filter_keys_by_value(mapping, filterBy)
    filter(k -> filterBy(mapping[k]), keys(mapping))
end

function load_header(sourceName = "$default_basedir/header")
    vec(readdlm("$sourceName"))
end

function load_data(sourceName = "$default_basedir/matrix")
    f = open("$sourceName")
    N = read(f, Int64)
    P = read(f, Int64)
    data = BitArray(N, P) # uninitialized

    # julia arrays are column major
    read!(f, data)
    close(f)

    data
end

function load_metadata(sourceName = "$default_basedir/metadata")
    f = open(sourceName)
    m = JSON.parse(f)
    close(f)
    m
end

# experiment -> target (for reverse = false)
function load_mapping(sourceName = "uniprot.mapping";
                      reverse = false)
    mappingMatrix = open(readcsv, sourceName)
    if reverse
        [mappingMatrix[i, 2] => mappingMatrix[i, 1] for i = 1:size(mappingMatrix)[1]]    
    else
        [mappingMatrix[i, 1] => mappingMatrix[i, 2] for i = 1:size(mappingMatrix)[1]]
    end
end

function load_pairs(sourceName = "biogrid_human_swissprot.csv")
    pairsMatrix = open(readcsv, sourceName)
    pairs = collect(zip(pairsMatrix[:, 1], pairsMatrix[:, 2]));
    reverse = map(p -> (p[2], p[1]), pairs)
    Set([pairs, reverse])
end

end
