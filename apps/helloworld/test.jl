docs = Vector{Vector{Tuple{Tuple{Int64, Int64}, Vector{Int32}}}}(300000)

for i = 1:300000
    doc = Vector{Tuple{Tuple{Int64, Int64}, Vector{Int32}}}(200)
    for j = 1:200
        doc[j] = ((i, j), Vector{Int32}(2))
    end
    docs[i] = doc
end

while true
end
