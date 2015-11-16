module Logging

export logln

function logln(message,
               indent = 0;
               show_time = true)

    head = join(["\t" for i=1:(indent)], "")

    if(show_time)
        t = strftime(time())
        head = string(t, " : \t", head)
    end

    println(head, message)
    flush(STDOUT)
end

end
