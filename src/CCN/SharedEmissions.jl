module SharedEmissions
export parallel_emissions!

using BaumWelchUtils
using ArrayViews

function interval_chunks(n, c)
    cs = Array(UnitRange{Int64}, c)
    over = n % c
    under_val = div(n, c)
    over_val = under_val + 1

    top_val = 0
    new_top_val = 0
    for i = 1:over
        new_top_val = top_val + over_val
        cs[i] = top_val+1:new_top_val
        top_val = new_top_val
    end

    for i = over+1:c
        new_top_val = top_val + under_val
        cs[i] = top_val+1:new_top_val
        top_val = new_top_val
    end

    cs
end

function fill_emissions!(to, buffer, data, interval, states, emission_dist!) 
    for si = 1:size(states, 1)
        state = states[si];
        emission_dist!(view(buffer, interval, si), state, view(data, :, interval))
    end     

    transpose!(view(to, :, interval), view(buffer, interval, :))
end

function parallel_emissions!(sh_data, sh_emissions, sh_emission_buffer, states, emission_dist!)
    n = size(sh_data, 2)
    wks = workers()
    nc = interval_chunks(n, size(wks, 1));

    rcs = [remotecall(wks[i],
                      fill_emissions!,
                      sh_emissions,
                      sh_emission_buffer,
                      sh_data,
                      nc[i],
                      states,
                      emission_dist!)
           for i = 1:nworkers()]

    map(wait, rcs)

    sh_emissions
end

end





