const data_path = "/Users/shijiewu/github/Research/20news.dat.200"
const vocab_size = 60057
const num_topics = 400
const alpha = 0.1
const beta = 0.1
const num_iterations = 10

function parse_line(index::Int64, line::AbstractString)::Vector{Tuple{Tuple{Int64, Int64},
                                                                      Int64}
                                                                }
    line_word_counts = Vector{Tuple{Tuple{Int64, Int64}, Int64}}()
    tokens = split(strip(line),  ' ')
    word_count_dict = Dict{Int64, Int64}()
    for token in tokens[2:end]
        word_count = split(token, ":")
        word_id = parse(Int64, word_count[1]) + 1
        if word_id in keys(word_count_dict)
            word_count_dict[word_id] += 1
        else
            word_count_dict[word_id] = 1
        end
    end
    for (word_id, count) in word_count_dict
        push!(line_word_counts, ((index, word_id), count))
    end
    return line_word_counts
end

function load_data(path::AbstractString)::Vector{Vector{Tuple{Tuple{Int64, Int64},
                                                              Int64}
                                                        }
                                                 }
    docs = Vector{Vector{Tuple{Tuple{Int64, Int64}, Int64}}}()
    num_lines::Int64 = 1
    open(path) do dataf
        for line in eachline(dataf)
            doc = parse_line(num_lines, line)
            push!(docs, doc)
            num_lines += 1
        end
    end
    return docs
end


function logDirichlet_vector(vector::Vector{Float64})::Float64
    sumLogGamma = 0.0
    logSumGamma = 0.0
    for value in vector
        sumLogGamma += lgamma(value)
        logSumGamma += value
    end
    sumLogGamma - lgamma(logSumGamma)
end

function logDirichlet_const(alpha::Float64, k::Int)::Float64
    k * lgamma(alpha) - lgamma(k * alpha)
end

function getColumn(word_topic_table::Vector{Dict{Int32,UInt64}}, topic::Int)::Vector{Float64}
    [get(word_topic_table[w], topic, 0.0) for w in 1:vocab_size]
end

function getRow(doc_topic_table::Vector{Dict{Int32,UInt64}}, doc_id::Int)::Vector{Float64}
    [get(doc_topic_table[doc_id], topic, 0.0) for topic in 1:num_topics]
end

function getLogLikelihood(word_topic_table::Vector{Dict{Int32,UInt64}}, doc_topic_table::Vector{Dict{Int32,UInt64}})::Float64
    lik::Float64 = 0.0
    for k = 1:num_topics
        temp = getColumn(word_topic_table, k)
        for w = 1:vocab_size
            temp[w] += beta
        end
        lik += logDirichlet_vector(temp)
        lik -= logDirichlet_const(beta, vocab_size)
    end

    for d = 1:num_docs
        temp = getRow(doc_topic_table, d)
        for k = 1:num_topics
            temp[k] += alpha
        end
        lik += logDirichlet_vector(temp)
        lik -= logDirichlet_const(alpha, num_topics)
    end
    return lik
end

println("loading data")
docs = load_data(data_path)
num_docs = length(docs)
topic_summary = [0 for i = 1:num_topics]
word_topic_table = [Dict{Int32, UInt64}() for w = 1:vocab_size]
doc_topic_table = [Dict{Int32, UInt64}() for d = 1:num_docs]
doc_topic_assignmnts = Vector{Vector{Int32}}()

println("initialization")
for doc_id in eachindex(docs)
    doc = docs[doc_id]
    doc_topic_assign_vec = Vector{Int64}()
    for token_count in doc
        word_id = token_count[1][2]
        count = token_count[2]
        for c = 1:count
            topic = rand(1:num_topics)
            push!(doc_topic_assign_vec, topic)
            if topic in keys(word_topic_table[word_id])
                word_topic_table[word_id][topic] += 1
            else
                word_topic_table[word_id][topic] = 1
            end

            if topic in keys(doc_topic_table[doc_id])
                doc_topic_table[doc_id][topic] += 1
            else
                doc_topic_table[doc_id][topic] = 1
            end
            topic_summary[topic] += 1
        end
    end
    push!(doc_topic_assignmnts, doc_topic_assign_vec)
end

word_topic_sum = [0 for word = 1:vocab_size]
for word_id = 1:vocab_size
    word_topic_dict = word_topic_table[word_id]
    word_topic_sum[word_id] = sum(values(word_topic_dict))
end

doc_topic_sum = [0 for doc_id = 1:num_docs]
for doc_id = 1:num_docs
    doc_topic_dict = doc_topic_table[doc_id]
    doc_topic_sum[doc_id] = sum(values(doc_topic_dict))
end

topic_sum = sum(topic_summary[topic] for topic = 1:num_topics)

type TopicCountPair
    topic::Int32
    count::UInt64
end

println("initialization done")

llh = [0.0 for iteration = 1:num_iterations]
sec = [0.0 for iteration = 1:num_iterations]
total_sec = 0.0

alpha_beta = alpha * beta
beta_sum = beta * vocab_size

@time for iteration = 1:num_iterations
    tic()
    println("iteration = ", iteration)
    for doc_id in eachindex(docs)
        topic_assignmnt_vec = doc_topic_assignmnts[doc_id]
        doc_topic_dict = doc_topic_table[doc_id]
        s_sum = 0
        r_sum = 0
        q_coeff = Vector{Float32}(num_topics)
        for t in 1:num_topics
            denom = topic_summary[t] + beta_sum
            s_sum += alpha_beta / denom
            if t in keys(doc_topic_dict)
                r_sum += (doc_topic_dict[t] * beta) / denom
                q_coeff[t] = (alpha + doc_topic_dict[t]) / denom
            else
                q_coeff[t] = alpha / denom
            end
        end

        doc = docs[doc_id]
        word_index = 1
        for token_count in doc
            word_id = token_count[1][2]
            count = token_count[2]
            word_topic_dict = word_topic_table[word_id]

            for c = 1:count
                old_topic = topic_assignmnt_vec[word_index]
                denom = topic_summary[old_topic] + beta_sum
                s_sum -= alpha_beta / denom
                s_sum += alpha_beta / (denom - 1)

                r_sum -= (doc_topic_dict[old_topic] * beta) / denom
                r_sum += ((doc_topic_dict[old_topic] - 1)* beta) / (denom - 1)
                q_coeff[old_topic] = (alpha + doc_topic_dict[old_topic] - 1) / (denom - 1)

                word_topic_dict[old_topic] -= 1
                if word_topic_dict[old_topic] == 0
                    delete!(word_topic_dict, old_topic)
                end
                doc_topic_dict[old_topic] -= 1
                if doc_topic_dict[old_topic] == 0
                    delete!(doc_topic_dict, old_topic)
                end
                topic_summary[old_topic] -= 1
                q_sum = 0
                q_terms = Vector{Tuple{Float32, Int32}}(length(word_topic_dict))
                index = 1
                for (topic, topic_count) in word_topic_dict
                    q_term = topic == old_topic ? q_coeff[topic] * (topic_count - 1) :
                        q_coeff[topic] * topic_count
                    q_terms[index] = (q_term, topic)
                    q_sum += q_term
                    index += 1
                end
                total_mass = q_sum + s_sum + r_sum
                sample = rand() * total_mass
                new_topic = 0
                if sample < q_sum
                    for q_term in q_terms
                        sample -= q_term[1]
                        if sample < 0
                            new_topic = q_term[2]
                        end
                    end
                elseif sample < q_sum + r_sum
                    sample -= q_sum
                    sample /= beta

                    for (topic, count) in doc_topic_dict
                        sample -= count / (topic_summary[topic] + beta_sum)
                        if sample < 0
                            new_topic = topic
                        end
                    end
                elseif sample < q_sum + r_sum + s_sum
                    sample -= q_sum + r_sum
                    sample /= alpha_beta
                    for k in 1:num_topics
                        sample -= 1.0 / (topic_summary[k] + beta_sum)
                        if sample < 0
                            new_topic = k
                        end
                    end
                else
                    error("sample = ", sample, " total_mass = ", total_mass)
                end
                denom = topic_summary[new_topic] + beta_sum
                s_sum -= (alpha_beta) / denom
                s_sum += alpha_beta / (denom + 1)
                if new_topic in keys(doc_topic_dict)
                    r_sum -= (doc_topic_dict[new_topic] * beta) / denom
                    r_sum += ((doc_topic_dict[new_topic] + 1) * beta) / (denom + 1)
                    q_coeff[new_topic] = (alpha + doc_topic_dict[new_topic] + 1) / (denom + 1)
                    doc_topic_dict[new_topic] += 1
                else
                    r_sum += beta / denom
                    q_coeff[new_topic] = (alpha + 1) / (denom + 1)
                    doc_topic_dict[new_topic] = 1
                end
                if new_topic in keys(word_topic_dict)
                    word_topic_dict[new_topic] += 1
                else
                    word_topic_dict[new_topic] = 1
                end
                topic_summary[new_topic] += 1
                topic_assignmnt_vec[word_index] = new_topic
                word_index += 1

                word_topic_sum_inline = sum(values(word_topic_dict))
                @assert word_topic_sum_inline == word_topic_sum[word_id]

                doc_topic_sum_inline = sum(values(doc_topic_dict))
                @assert doc_topic_sum_inline == doc_topic_sum[doc_id]

                topic_sum_inline = sum(topic_summary[topic] for topic = 1:num_topics)
                @assert topic_sum_inline == topic_sum
            end # sampling for this token done
        end # sampling for this word done
    end # sampling for this doc done
    sec[iteration] = toc()
    llh[iteration] = getLogLikelihood(word_topic_table, doc_topic_table)
    println("Client completed iteration $(iteration)")
    println("    Elapsed seconds: $(sec[iteration])")
    println("    Log-likelihood: $(llh[iteration])")
end # iteration done
