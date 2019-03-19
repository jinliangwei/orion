const data_path = "/proj/BigLearning/jinlianw/data/clueweb.libsvm.25M"

function parse_line(line::AbstractString)::Vector{Tuple{Int64,
                                                        Int64}
                                                  }

    line_word_counts = Vector{Tuple{Int64, Int64}}()
    tokens = split(strip(line),  ' ')
    for token in tokens[2:end]
        word_count_tokens = split(token, ":")
        word_id = parse(Int64, word_count_tokens[1])
        word_count = parse(Int64, word_count_tokens[2])
        push!(line_word_counts, (word_id, word_count))
    end
    return line_word_counts
end

function load_data(path::AbstractString)::Vector{Vector{Tuple{Int64,
                                                              Int64}
                                                        }
                                                 }
    docs = Vector{Vector{Tuple{Int64, Int64}}}()
    open(path) do dataf
        for line in eachline(dataf)
            doc = parse_line(line)
            push!(docs, doc)
        end
    end
    return docs
end

function get_max_word_id(docs::Vector{Vector{Tuple{Int64, Int64}}})::Int64
    max_word_id = 0
    for doc in docs
        for word_count_pair in doc
            word_id = word_count_pair[1]
            max_word_id = max(max_word_id, word_id)
        end
    end
    return max_word_id
end

function shuffle_words(docs::Vector{Vector{Tuple{Int64, Int64}}}, word_id_perm)
    for doc_idx in eachindex(docs)
        doc = docs[doc_idx]
        new_doc = Vector{Tuple{Int64, Int64}}()
        for (word_id, count) in doc
            new_word_id = word_id_perm[word_id + 1]
            push!(new_doc, (new_word_id, count))
        end
        docs[doc_idx] = new_doc
    end
end

function output_data(docs, path)
    perm_fobj = open(path * ".perm", "w")

    for doc_idx in eachindex(docs)
        doc = docs[doc_idx]
        write(perm_fobj, string(doc_idx) * " ")
        for (word_id, count) in doc
            write(perm_fobj, string(word_id) * ":" * string(count) * " ")
        end
        write(perm_fobj, "\n")
    end
    close(perm_fobj)
end

println("loading data")
docs = load_data(data_path)
num_docs = length(docs)
println("num_docs = ", num_docs)
max_word_id = get_max_word_id(docs)

word_id_perm = [i for i = 0:(max_word_id + 1)]
shuffle!(word_id_perm)
shuffle!(docs)
println("shuffling words")
shuffle_words(docs, word_id_perm)
println("output data")
output_data(docs, data_path)
