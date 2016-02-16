module SimpleEnrichment

using Compat

using ProgressMeter
using PureSeq

function get_segway_file(cellType, colorMap)
    root = "http://ngs.sanger.ac.uk/production/ensembl/regulation/hg38/projected_segmentations"
    mkpath("data/segway")
    run(`wget $root/$cellType.bb -q -O data/segway/$cellType.bb`)
    run(`bigBedToBed data/segway/$cellType.bb data/segway/$cellType.bed`)
    for color in keys(colorMap)
        @compat run(pipeline(
            "data/segway/$cellType.bed",
            ignorestatus(`grep $color`),
            "data/segway/$(cellType)_$(colorMap[color]).bed"
        ))
    end
end

segwayColorMap = [
    "10,190,254" => "ctcf",
    "209,157,0" => "tfbs",
    "225,225,225" => "inactive",
    "250,202,0" => "enhancer",
    "255,0,0" => "promoter",
    "255,105,105" => "promoter_flank",
    "255,252,4" => "open"
]

segwayCellTypes = [
    "A549","DND-41","GM12878","H1ESC","HeLa-S3","HepG2",
    "HMEC","HSMM","HSMMtube","HUVEC","IMR90","K562",
    "Monocytes-CD14+","Nha","NHDF-AD","NHEK","Nhlf","Osteobl"
]

@showprogress for t in segwayCellTypes
    get_segway_file(t, segwayColorMap)
end

end
