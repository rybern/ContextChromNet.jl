module SynthResults

function display_ticks(vars, ticks)
    for (vari, var) = enumerate(vars)
        println("$vari | $var:")
        for (ticki, tick) = enumerate(ticks[vari])
            println("\t$ticki | $tick")
        end
    end
end

# Typical output:
#
#
# model_optimizer (1):
# 	Full Covariance BW (1)
# 	True model (2)
# data_generator (2):
# 	Generating density 1.0 (1)
# 	Generating density 0.5 (2)
# 	Generating density 0.1 (3)
# validation_measure (3):
# 	Label Accuracy (1)
# 	Edge Accuracy (2)
# 	Test Log-Likelihood (3)
# 	Network Enrichment (4)
# iteration (4):
# 	1 (1)
# 	2 (2)
# 	3 (3)
# 	4 (4)
# 	5 (5)


end
