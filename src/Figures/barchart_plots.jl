module barchart_plots

export grouped_bar_plot, basic_bar_plot

using PyPlot
plt = PyPlot

# bars will be [[bar_type1_means, bar_type1_stds, bar_type1_label], ...]
function grouped_bar_plot(bars, group_labels, xaxis, yaxis, title;
                          fsize = 40,
                          bar_width = 0.35,
                          err_width = 11,
                          opacity = 0.4,
                          dumpfile = false)
    fig, ax = plt.subplots()

    index = [1:length(bars[1][1])]

    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

    for i = 1:length(bars)
        (means, stds, label) = bars[i]

        rects1 = plt.bar(index + bar_width * (i-1), means, bar_width,
                         alpha=opacity,
                         color=colors[i],
                         yerr=stds,
                         error_kw=Dict(["lw", "capsize", "capthick", "ecolor"], [err_width, err_width * 2, err_width/3, "black"]),
                         label=label)
    end

    plt.xlabel(xaxis, fontsize=fsize)
    plt.ylabel(yaxis, fontsize=fsize)
    plt.title(title, fontsize = fsize*1.1)
    plt.yticks(fontsize = fsize*.85)
    plt.xticks(index + bar_width, group_labels, fontsize = fsize*.9)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize = fsize)

    plt.tight_layout()

    if(dumpfile == false)
        plt.show()
    else
        savefig(dumpfile, bbox_inches="tight")
    end
end

function basic_bar_plot(bars, labels, xaxis, yaxis, title;
                        fsize = 40,
                        bar_width = 1,
                        opacity = 0.4,
                        dumpfile = false)

    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
    println(typeof(bars))
    println(typeof(bars[1]))
    println(typeof(labels))
    println(typeof(labels[1]))

    rects = plt.bar(1:length(bars),
                     bars, 
                     bar_width,
                     alpha=opacity)
                     color=colors,
                   #  label=label)

#    for i = 1:length(rects)
#        rects[i].set_color(colors[i])
#    end

    plt.xlabel(xaxis, fontsize=fsize)
    plt.ylabel(yaxis, fontsize=fsize)
    plt.title(title, fontsize = fsize*1.1)
    plt.yticks(fontsize = fsize*.85)

    plt.ylim(1, 3.5)

    plt.xticks([1:length(bars)] + bar_width/2.0, labels, fontsize = fsize * .85)
    
    plt.legend(fontsize = fsize)

    plt.tight_layout()

    if(dumpfile == false)
        plt.show()
    else
        savefig(dumpfile, bbox_inches="tight")
    end
end


end
