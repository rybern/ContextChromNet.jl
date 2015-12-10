module SimpleLogging

export logstr, logstrln

function logstrln(message :: String,
                  indent :: Union(Type{Nothing}, Integer) = 0,
                  show_time = true)
    logstr(string(message, "\n"), indent, show_time)
end

function logstr(message :: String,
                indent :: Union(Type{Nothing}, Integer) = 0,
                show_time = true)
    if indent == Nothing
        return
    end

    head = join(["\t" for i=1:(indent)], "")

    if(show_time)
        t = strftime(time())
        head = string(t, " : \t", head)
    end

    print(head, message)
    flush(STDOUT)
end

end
