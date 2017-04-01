function parse_line(line::AbstractString)
    tokens = split(line, ',')
    @assert length(tokens) == 3
    key_array = [parse(Int64, AbstractString(tokens[1])),
                 parse(Int64, AbstractString(tokens[2]))]
    value = parse(Float64, AbstractString(tokens[3]))
    return (key_array, value_tuple)
end

function my_map(f::Function)
    println(code_typed(f),())
end

my_map(parse_line)
