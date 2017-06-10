module OrionWorker

function set_lib_path(path::AbstractString)
    global const lib_path = path
end

function helloworld()
    ccall((:orion_helloworld, lib_path), Void, ())
end

end
