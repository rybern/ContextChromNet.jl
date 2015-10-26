#samples = ["chromhmm_sample_binary.txt", "chromhmm_sample_binary1.txt", "chromhmm_sample_binary2.txt", "chromhmm_sample_binary3.txt"]

function join_runs(runs)
    celltypes = unique([run[1] for run = runs])
    total = Array(Array{Array{Bool, 2}, 1}, size(celltypes, 1));

    for i = 1:size(celltypes, 1)
        celltype_runs = filter(run -> run[1] == celltypes[i], runs)

        
        sort_by_chr = sort(celltype_runs, by = run -> run[2])

        total[i] = [run[4] for run = sort_by_chr]

        println("celltype $(celltypes[i]) has $(length(total[i])) files with lengths $([size(data, 2) for data = total[i]]) totalling $(sum([size(data, 2) for data = total[i]]))")
    end

    collected = reduce(hcat, reduce(vcat, total))
    binary = convert(BitArray, collected)

    (celltypes, runs[1][3], binary)
end

function parse_bin_file(filepath)
    raw = readdlm(filepath, '\t')
    celltype = raw[1,1]
    chrN = raw[1,2][4:end]
    chr = 0
    if chrN == "X"
        chr = 100
    elseif chrN == "Y" 
        chr = 101
    else
        chr = int(chrN)
    end
    header = vec(raw[2,:])
    data = transpose(bool(raw[3:end, :]))

    println("read file $filepath, had celltype $celltype, $(size(data, 2)) lines")
    
    (celltype, chr, header, data)
end

function save_bin_data(outdir, celltypes, header, data)
    if(!isdir(outdir))
        mkdir(outdir)
    end

    open(s -> serialize(s, data), "$outdir/data.serial", "w")
    save_arr("$outdir/header", header)
    save_arr("$outdir/celltypes", celltypes)

    nothing
end

function save_arr(outfile, arr)
    s = open(outfile, "w")

    for x = arr
        write(s, x, "\n")
    end

    close(s)
end

    
