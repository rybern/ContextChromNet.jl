module BarchartPlots

export grouped_bar_plot, basic_bar_plot

using PyPlot
plt = PyPlot

toy_bars = {{[2.389051612300086,1.6727820069204156,0.9078551882757762,0.9995998247653397,0.9864007928206602],[4.681111291435602e-16,2.340555645717801e-16,0.0,0.0,1.1702778228589004e-16],"GLasso"},
            {[6.517806239370364,3.5684366782006918,2.15877605884945,1.4408678435057714,1.0438351022231873],[0.03983696789005869,0.020800276511351776,0.19714568799941457,0.005120971354378308,0.0024783485074209776],"Full Cov"}, 
{[7.379351240339682,4.19621937716263,2.9221130414242142,1.7951878891791258,1.2567953913154088],[0.03155344430613882,0.030815852105142382,0.035209762207639134,0.0049632122090618215,0.0016091111152237354],"Truth"}
,{[7.789538501425978,4.429411764705882,3.1036242083040118,1.9248686731153124,1.2991678464602214],[0.0,9.362222582871203e-16,4.681111291435602e-16,0.0,2.340555645717801e-16],"Diag Cov"}}

function toy()
    grouped_bar_plot(toy_bars, map(string, [1:5]), "x axis", "y axis", "title", y_limits = true)
end

# bars will be [[bar_type1_means, bar_type1_stds, bar_type1_label], ...]
function grouped_bar_plot(bars,
                          group_labels,
                          xaxis,
                          yaxis,
                          title;
                          fsize = 40,
                          bar_width = 0.35,
                          err_width = 11,
                          opacity = 0.4,
                          y_limits = false,
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

    if y_limits
        y = reduce(vcat, [bar[1] for bar = bars])
        y_min = minimum(y)
        y_max = maximum(y)
        y_range = y_max - y_min
        buffer = y_range / 20;

        y_lower = y_min - buffer
        y_upper = y_max + buffer

        plt.ylim([y_lower, y_upper])
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
