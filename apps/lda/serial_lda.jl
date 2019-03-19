const data_path = "/proj/BigLearning/jinlianw/data/nytimes.dat.perm"
const num_topics = 1000
const alpha = Float32(0.1)
const beta = Float32(0.1)
const alpha_beta = alpha * beta
const num_iterations = 100

num_tokens = 0
num_dist_tokens = 0
num_docs = UInt64(0)
vocab_size = 0
max_token_count_per_doc_word = 0

function parse_line(line_num::Int64, line::AbstractString,
                    token_keys::Vector{Tuple{Int64, Int64, Int64}},
                    token_topics::Vector{Int32})
    # Parse each line to be word counts
    tokens = split(strip(line),  ' ')
    idx = 1
    global num_docs += 1
    global vocab_size
    global max_token_count_per_doc_word
    for token in tokens[2:end]
        word_count = split(token, ":")
        word_id = parse(Int64, word_count[1]) + 1
        if word_id > vocab_size
            vocab_size = word_id
        end
        count = parse(Int32, word_count[2])
        global num_tokens += count
        global num_dist_tokens += 1
        if count > max_token_count_per_doc_word
            max_token_count_per_doc_word = count
        end
        for c = 1:count
            push!(token_keys, (c, word_id, line_num))
            push!(token_topics, 0)
        end
        idx += 1
    end
    return token_keys, token_topics
end

function load_data(path::AbstractString)
    token_keys = Vector{Tuple{Int64, Int64, Int64}}()
    token_topics = Vector{Int32}()
    num_lines::Int64 = 1
    open(path) do dataf
        for line in eachline(dataf)
            doc = parse_line(num_lines, line, token_keys, token_topics)
            num_lines += 1
        end
    end
    return (token_keys, token_topics)
end

function initialize(topic_assignments::Tuple{Vector{Tuple{Int64, Int64, Int64}},
                                             Vector{Int32}},
                    word_topic_dict_table::Vector{Dict{Int32, UInt64}},
                    doc_topic_table::Vector{Dict{Int32, UInt64}},
                    topic_summary::Vector{UInt64}
                    )
    println("initialization")
    topic_assignment_keys = topic_assignments[1]
    topic_assignment_topics = topic_assignments[2]

    for idx in eachindex(topic_assignment_keys)
        doc_id = topic_assignment_keys[idx][3]
        word_id = topic_assignment_keys[idx][2]
        word_topic_dict = word_topic_dict_table[word_id]
        doc_topic_dict = doc_topic_table[doc_id]

        topic = rand(1:num_topics)
        topic_assignment_topics[idx] = topic
        word_topic_count = get(word_topic_dict, topic, 0)
        word_topic_dict[topic] = word_topic_count + 1
        doc_topic_count = get(doc_topic_dict, topic, 0)
        doc_topic_dict[topic] = doc_topic_count + 1
        topic_summary[topic] += 1
    end
end

function create_word_topic_vec(word_topic_dict::Dict{Int32, UInt64}
                               )::Tuple{Vector{UInt64}, Vector{Int32}}
    num_nonzero_counts = length(word_topic_dict)
    word_topic_count_vec = Vector{UInt64}(num_nonzero_counts)
    word_topic_index_vec = Vector{Tuple{Int32, Int32}}(num_nonzero_counts)
    index = 1
    for (topic, topic_count) in word_topic_dict
        word_topic_count_vec[index] = topic_count
        word_topic_index_vec[index] = (topic, index)
        index += 1
    end
    @assert index == length(word_topic_index_vec) + 1
    sort!(word_topic_index_vec, by = x -> word_topic_count_vec[x[2]], rev = true)
    sort!(word_topic_count_vec, rev = true)
    word_topic_vec = map(x -> x[1], word_topic_index_vec)
    return (word_topic_count_vec, word_topic_vec)
end

function compute_r_sum(doc_topic_table::Vector{Dict{Int32, UInt64}},
                       num_docs::UInt64,
                       beta_sum)
    r_sum = zeros(num_docs)
    for doc_id in eachindex(doc_topic_table)
        doc_topic_dict = doc_topic_table[doc_id]
        doc_r_sum = 0.0
        for topic_count in doc_topic_dict
            topic = topic_count[1]
            count = topic_count[2]
            doc_r_sum += count * beta / (beta_sum + topic_summary[topic])
        end
        r_sum[doc_id] = doc_r_sum
    end
    return r_sum
end

function compute_s_sum(topic_summary::Vector{UInt64},
                       alpha_beta, beta_sum)
    s_sum = Float32(0.0)
    for topic_count in topic_summary
        s_sum += alpha_beta / (beta_sum + topic_count)
    end
    return s_sum
end

function compute_q_coeff(doc_topic_table::Vector{Dict{Int32, UInt64}},
                         topic_summary::Vector{UInt64},
                         num_topics, num_docs,
                         alpha, beta_sum)
    q_coeff = [zeros(Float32, num_topics) for i = 1:num_docs]
    for doc_id in eachindex(doc_topic_table)
        doc_topic_dict = doc_topic_table[doc_id]
        q_coeff_vec = q_coeff[doc_id]
        for topic = 1:num_topics
            count = get(doc_topic_dict, topic, 0)
            q_coeff_vec[topic] = (alpha + count) / (beta_sum + topic_summary[topic])
        end
    end
end

function find_index(vec::Vector{Int32}, key)
    for idx in eachindex(vec)
        element = vec[idx]
        if key == element
            return idx
        end
    end
    return -1
end

function sample_all_words(topic_assignments,
                          doc_topic_table,
                          word_topic_vec_table,
                          topic_summary,
                          s_sum,
                          r_sum,
                          q_coeff,
                          alpha,
                          alpha_beta,
                          beta,
                          num_topics,
                          beta_sum,
                          q_terms_val,
                          q_terms_topic,
                          iteration)

    topic_assignment_keys = topic_assignments[1]
    topic_assignment_topics = topic_assignments[2]
    old_word_id = Int64(-1)
    old_doc_id = Int64(-1)
    q_sum = Float32(0.0)
    num_nonzero_q_terms = UInt64(0)
    topic_assignment_key = Vector{Int64}(3)

    for topic_assignment_idx in eachindex(topic_assignment_keys)
        key = topic_assignment_keys[topic_assignment_idx]
        from_int64_to_keys(key, dims, topic_assignment_key)
        old_topic = topic_assignment_topics[topic_assignment_idx]

        doc_id = topic_assignment_key[3]
        word_id = topic_assignment_key[2]

        doc_topic_dict = doc_topic_table[doc_id]
        word_topic_vec_pair = word_topic_vec_table[word_id]
        word_topic_count_vec = word_topic_vec_pair[1]
        word_topic_vec = word_topic_vec_pair[2]
        doc_q_coeff = q_coeff[doc_id]

        if old_word_id != word_id ||
            old_doc_id != doc_id
            q_sum = 0.0f0
            num_nonzero_q_terms = UInt64(0)
            for index in eachindex(word_topic_vec)
                topic_count = word_topic_count_vec[index]
                topic = word_topic_vec[index]
                q_term = doc_q_coeff[topic] * topic_count
                num_nonzero_q_terms += 1
                q_terms_val[num_nonzero_q_terms] = q_term
                q_terms_topic[num_nonzero_q_terms] = topic
                q_sum += q_term
            end
            old_word_id = word_id
            old_doc_id = doc_id
        end

        @assert q_sum >= 0
        denom = topic_summary[old_topic] + beta_sum
        s_sum -= alpha_beta / denom
        s_sum += alpha_beta / (denom - 1)
        r_sum[doc_id] -= (doc_topic_dict[old_topic] * beta) / denom
        r_sum[doc_id] += ((doc_topic_dict[old_topic]- 1) * beta) / (denom - 1)
        doc_q_coeff[old_topic] = (alpha + doc_topic_dict[old_topic] - 1) / (denom - 1)
        doc_r_sum = r_sum[doc_id]
        doc_topic_dict[old_topic] -= 1
        if doc_topic_dict[old_topic] == 0
            delete!(doc_topic_dict, old_topic)
        end

        old_topic_index = find_index(word_topic_vec, old_topic)
        old_topic_count = word_topic_count_vec[old_topic_index]
        word_topic_count_vec[old_topic_index] = old_topic_count - 1
        q_sum -= q_terms_val[old_topic_index]
        q_terms_val[old_topic_index] = (old_topic_count - 1) * doc_q_coeff[old_topic]
        q_sum += q_terms_val[old_topic_index]
        topic_summary[old_topic] -= 1

        total_mass = q_sum + doc_r_sum + s_sum
        sample = rand() * total_mass
        new_topic::Int32 = Int32(0)
        if sample < q_sum
            for idx in 1:num_nonzero_q_terms
                sample -= q_terms_val[idx]
                if sample < 0
                    new_topic = q_terms_topic[idx]
                    break
                end
            end
            if sample >= 0
                new_topic = q_terms_topic[num_nonzero_q_terms]
            end
        elseif sample < q_sum + doc_r_sum
            sample -= q_sum
            sample /= beta
            topic_last = Int32(1)
            for (topic, topic_count) in doc_topic_dict
                sample -= topic_count / (topic_summary[topic] + beta_sum)
                topic_last = topic
            if sample < 0
                new_topic = topic
                break
            end
            end
            if sample >= 0
                new_topic = topic_last
            end
        elseif sample <= q_sum + doc_r_sum + s_sum
            sample -= q_sum + doc_r_sum
            sample /= alpha_beta
            for t in 1:num_topics
                sample -= 1.0 / (topic_summary[t] + beta_sum)
                if sample < 0
                    new_topic = Int32(t)
                    break
                end
            end
            if sample >= 0
                new_topic = Int32(num_topics)
            end
        else
            error("sample = ", sample, " total_mass = ", total_mass)
        end

        denom = topic_summary[new_topic] + beta_sum
        s_sum -= alpha_beta / denom
        s_sum += alpha_beta / (denom + 1)
        if haskey(doc_topic_dict, new_topic)
            r_sum[doc_id] -= (doc_topic_dict[new_topic] * beta) / denom
            r_sum[doc_id] += ((doc_topic_dict[new_topic] + 1) * beta) / (denom + 1)
            doc_q_coeff[new_topic] = (alpha + doc_topic_dict[new_topic] + 1) / (denom + 1)
            doc_topic_dict[new_topic] += 1
        else
            r_sum[doc_id] += beta / denom
            doc_q_coeff[new_topic] = (alpha + 1) / (denom + 1)
            doc_topic_dict[new_topic] = 1
        end

        new_topic_index = find_index(word_topic_vec, new_topic)
        if new_topic_index != -1
            new_topic_count = word_topic_count_vec[new_topic_index] + 1
            word_topic_count_vec[new_topic_index] = new_topic_count
            q_sum -= q_terms_val[new_topic_index]
        else
            push!(word_topic_vec, new_topic)
            push!(word_topic_count_vec, 1)
            new_topic_count = UInt64(1)
            new_topic_index = length(word_topic_vec)
            num_nonzero_q_terms += 1
        end
        println("doc_id = ", doc_id, " word_id = ", word_id, " old_topic = ", old_topic,
                " new_topic = ", new_topic)
        q_terms_val[new_topic_index] = new_topic_count * doc_q_coeff[new_topic]
        q_terms_topic[new_topic_index] = new_topic
        q_sum += q_terms_val[new_topic_index]

        topic_summary[new_topic] += 1
        topic_assignment_topics[topic_assignment_idx] = new_topic
    end
    return s_sum
end

function sort_word_topic_vecs(word_topic_vec_table::Vector{Tuple{Vector{UInt64}, Vector{Int32}}})
    for word_topic_vec_idx in eachindex(word_topic_vec_table)
        word_topic_count_vec = word_topic_vec_table[word_topic_vec_idx][1]
        word_topic_vec = word_topic_vec_table[word_topic_vec_idx][2]
        word_topic_index_vec = Vector{Tuple{Int32, Int32}}(length(word_topic_vec))
        @assert length(word_topic_count_vec) == length(word_topic_vec)
        for idx in eachindex(word_topic_vec)
            word_topic_index_vec[idx] = (word_topic_vec[idx], idx)
        end
        sort!(word_topic_index_vec, by = x -> word_topic_count_vec[x[2]], rev = true)
        sort!(word_topic_count_vec, rev = true)
        word_topic_vec = map(x -> x[1], word_topic_index_vec)

        num_nonzeros = 0
        for topic_count in word_topic_count_vec
            if topic_count == 0
                break
            end
            num_nonzeros += 1
        end
        resize!(word_topic_count_vec, num_nonzeros)
        resize!(word_topic_vec, num_nonzeros)
        @assert length(word_topic_count_vec) == length(word_topic_vec)
        word_topic_vec_table[word_topic_vec_idx] = (word_topic_count_vec, word_topic_vec)
    end
end

function compute_llh(word_topic_vec_table::Vector{Tuple{Vector{UInt64}, Vector{Int32}}},
                     docc_topic_table::Vector{Dict{Int32, UInt64}},
                     word_log_gamma_sum::Vector{Float32},
                     alpha,
                     beta,
                     num_topics)
    llh = 0.0
    for word_topic_vec_index in eachindex(word_topic_vec_table)
        word_topic_vec_pair = word_topic_vec_table[word_topic_vec_index]
        word_topic_count_vec = word_topic_vec_pair[1]
        word_topic_vec = word_topic_vec_pair[2]

        @assert length(word_topic_count_vec) == length(word_topic_vec)
        for idx in eachindex(word_topic_vec)
            topic = word_topic_vec[idx]
            count = word_topic_count_vec[idx]
            word_log_gamma_sum[topic] += lgamma(count + beta)
        end
        llh += (num_topics - length(word_topic_vec)) * lgamma(beta)
    end
    for topic in eachindex(word_log_gamma_sum)
        topic_log_gamma_sum_val = word_log_gamma_sum[topic]
        llh += topic_log_gamma_sum_val - lgamma(vocab_size * beta + topic_summary[topic])
        llh += lgamma(vocab_size * beta) - vocab_size * lgamma(beta)
        word_log_gamma_sum[topic] = 0.0
    end
    # compute topic likelihood
    for doc_topic_dict in doc_topic_table
        doc_log_gamma_sum = 0.0
        doc_total_word_count = 0
        for (topic, count) in doc_topic_dict
            doc_total_word_count += count
            doc_log_gamma_sum += lgamma(count + alpha)
        end
        doc_log_gamma_sum += (num_topics - length(doc_topic_dict)) * lgamma(alpha)
        llh += doc_log_gamma_sum - lgamma(alpha * num_topics + doc_total_word_count)
        llh += lgamma(num_topics * alpha) - num_topics * lgamma(alpha)
    end
    return llh
end

function train(topic_assignments,
               doc_topic_table,
               word_topic_vec_table,
               topic_summary,
               s_sum,
               r_sum,
               q_coeff,
               alpha,
               alpha_beta,
               beta,
               num_topics,
               beta_sum)
    q_terms_val = Vector{Float32}(num_topics)
    q_terms_topic = Vector{Int32}(num_topics)
    word_log_gamma_sum = zeros(Float32, num_topics)
    llh_vec = Vector{Float32}()
    time_vec = Vector{Float32}()
    start_time = now()
    last_time = start_time

    @time for iteration = 1:num_iterations
        @time sample_all_words(topic_assignments,
                               doc_topic_table,
                               word_topic_vec_table,
                               topic_summary,
                               s_sum,
                               r_sum,
                               q_coeff,
                               alpha,
                               alpha_beta,
                               beta,
                               num_topics,
                               beta_sum,
                               q_terms_val,
                               q_terms_topic,
                               iteration)


        sort_word_topic_vecs(word_topic_vec_table)
        llh = 0
        if iteration % 1 == 0 ||
            iteration == num_iterations
            llh = compute_llh(word_topic_vec_table,
                              doc_topic_table,
                              word_log_gamma_sum,
                              alpha,
                              beta,
                              num_topics)

            push!(llh_vec, llh)
            curr_time = now()
            diff_time = Int(Dates.value(curr_time - last_time)) / 1000
            last_time = curr_time
            elapsed = Int(Dates.value(curr_time - start_time)) / 1000
            push!(time_vec, elapsed)
            println("iteration = ", iteration, " elapsed = ", elapsed, " llh = ", llh)
        end
    end
    println(time_vec)
    println(llh_vec)
    return (time_vec, llh_vec)
end

function from_int64_to_keys(key::Int64, dims::Vector{Int64}, dim_keys::Vector{Int64})
    index = 1
    for dim in dims
        key_this_dim = key % dim + 1
        dim_keys[index] = key_this_dim
        key = fld(key, dim)
        index += 1
    end
end

function from_keys_to_int64(key, dims::Vector{Int64})::Int64
    key_int = 0
    for i = length(dims):-1:2
        key_int += key[i] - 1
        key_int *= dims[i - 1]
    end
    key_int += key[1] - 1
    return key_int
end

function convert_all_keys_to_int64(keys::Vector{Tuple{Int64, Int64, Int64}},
                                   dims::Vector{Int64})
    int_keys = Vector{Int64}(length(keys))
    for idx in eachindex(keys)
        key_tuple = keys[idx]
        int_keys[idx] = from_keys_to_int64(key_tuple, dims)
    end
    return int_keys
end


println("loading data")
topic_assignments = load_data(data_path)
println("num_tokens = ", num_tokens)
println("num_dist_tokens = ", num_dist_tokens)
println("num_docs is ", num_docs)

dims = [Int64(max_token_count_per_doc_word), Int64(vocab_size), Int64(num_docs)]

topic_summary = zeros(UInt64, num_topics)
word_topic_dict_table = [Dict{Int32, UInt64}() for w = 1:vocab_size]
doc_topic_table = [Dict{Int32, UInt64}() for d = 1:num_docs]

srand(1)
@time initialize(topic_assignments,
                 word_topic_dict_table,
                 doc_topic_table,
                 topic_summary)
println("initialization done")

word_topic_vec_table = map(create_word_topic_vec, word_topic_dict_table)
println("constructed word_topic_vec_table")

const beta_sum = beta * vocab_size
r_sum = compute_r_sum(doc_topic_table, num_docs, beta_sum)

s_sum = compute_s_sum(topic_summary, alpha_beta, beta_sum)

q_coeff = [zeros(Float32, num_topics) for i = 1:num_docs]

topic_assignments = (convert_all_keys_to_int64(topic_assignments[1], dims),
                     topic_assignments[2])

(time_vec, llh_vec) = train(topic_assignments,
                            doc_topic_table,
                            word_topic_vec_table,
                            topic_summary,
                            s_sum,
                            r_sum,
                            q_coeff,
                            alpha,
                            alpha_beta,
                            beta,
                            num_topics,
                            beta_sum)

llh_fobj = open("results.order/" * split(PROGRAM_FILE, "/")[end] * "-" *
                 split(data_path, "/")[end] * "-" * string(1) * "-" *
                 string(num_topics) * "-" * string(num_iterations) * "-" * string(now()) * ".loss", "w")
for idx in eachindex(time_vec)
    write(llh_fobj, string(idx) * "\t" * string(time_vec[idx]) * "\t" * string(llh_vec[idx]) * "\n")
end
close(llh_fobj)
