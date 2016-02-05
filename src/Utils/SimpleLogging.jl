module SimpleLogging

using Compat
export logstr, logstrln

@compat function logstrln(message :: String,
                  indent :: Union{Type{Void}, Integer} = 0,
                  show_time = true)
    logstr(string(message, "\n"), indent, show_time)
end

@compat function logstr(message :: String,
                indent :: Union{Type{Void}, Integer} = 0,
                show_time = true)
    if indent == Void
        return
    end

    head = join(["\t" for i=1:(indent)], "")

    if(show_time)
        @compat t = Libc.strftime(time())
        head = string(t, " : \t", head)
    end

    print(head, message)
    flush(STDOUT)
end

end
