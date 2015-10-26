include("utils.jl")
include("hmm_types.jl")
using utils
using hmm_types


chr_sizes = ["chr1"     248956422;
             "chr2"	242193529;
             "chr3"	198295559;
             "chr4"	190214555;
             "chr5"	181538259;
             "chr6"	170805979;
             "chr7"	159345973;
             "chr8"	145138636;
             "chr9"	138394717;
             "chr11"	135086622;
             "chr10"	133797422;
             "chr12"	133275309;
             "chr13"	114364328;
             "chr14"	107043718;
             "chr15"	101991189;
             "chr16"	90338345;
             "chr17"	83257441;
             "chr18"	80373285;
             "chr19"	58617616;
             "chr20"	64444167;
             "chr22"	50818468;
             "chr21"	46709983;
             "chrX"	156040895;
             "chrY"	57227415]

function rand_tracks(output, k = 15, bin_size = 1000)
    total_size = sum(chr_sizes[:, 2])

    n_bins = int(total_size / bin_size)
    track = rand(1:k, n_bins)
    bed_annotation(track, k, "rand_beds/")
end


function binary_to_bed(output, track, track_name, track_desc, bin_size)
    accum_chr_sizes = Array(Any, size(chr_sizes))

    running = 0
    for i = 1:size(chr_sizes, 1)
        running += chr_sizes[i, 2]
        accum_chr_sizes[i, 1] = chr_sizes[i, 1]
        accum_chr_sizes[i, 2] = running
    end

    write(output, "track name=\"$track_name\" description=\"$track_desc\"\n")
    total_size = accum_chr_sizes[end, 2]
    chr_index = 1

    offset = 0
    for pos = 0: int(total_size / bin_size -1)
        while (pos * bin_size > accum_chr_sizes[chr_index, 2])
            offset = accum_chr_sizes[chr_index, 2]
            chr_index += 1
        end

        if (track[pos+1] == 1)
            write(output, "$(accum_chr_sizes[chr_index,1])\t$(pos*bin_size-offset)\t$((pos+1)*bin_size - offset)\n")
        end
    end
end


function bed_annotation(labels :: Array{Int, 1}, k, outputDir)
    if(!isdir(outputDir))
        mkdir(outputDir)
    end

    for state = 1:k
        state_track = int(labels .== state)
        open(s -> binary_to_bed(s, state_track, "state $state", "state $state", 1000),
             "$outputDir/state$(state).bed", "w")
    end
end

function bed_annotation(estimate_dump, k, outputDir)
    estimate = open(deserialize, estimate_dump)

    labels = utils.gammaAssignments(estimate.gamma)
        
    for state = 1:k
        state_track = int(labels .== state)
        open(s -> binary_to_bed(s, state_track, "state $state", "state $state", 1000),
             "$outputDir/state$(state).bed", "w")
    end
end





