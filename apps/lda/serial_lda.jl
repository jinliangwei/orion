const data_path = "/proj/BigLearning/jinlianw/data/nytimes.dat.perm.5K"
const vocab_size = 110000
const num_topics = 100
const alpha = 0.1
const beta = 0.1
const alpha_beta = alpha * beta
const num_iterations = 2

num_tokens = 0
num_dist_tokens = 0
num_docs = 0

function parse_line(line_num::Int64, line::AbstractString)::Vector{Tuple{Tuple{Int64, Int64, Int64},
                                                                      Int32}
                                                                }
    # Parse each line to be word counts
    tokens = split(strip(line),  ' ')
    token_vec = Vector{Tuple{Tuple{Int64, Int64, Int64},
                            Int32}
                      }()
    idx = 1
    global num_docs += 1
    for token in tokens[2:end]
        word_count = split(token, ":")
        word_id = parse(Int64, word_count[1]) + 1
        count = parse(Int32, word_count[2])
        global num_tokens += count
        global num_dist_tokens += 1
        for c = 1:count
            push!(token_vec, ((line_num, word_id, c), 0))
        end
        idx += 1
    end
    return token_vec
end

function load_data(path::AbstractString)::Vector{Tuple{Tuple{Int64, Int64, Int64},
                                                       Int32}
                                                 }
    docs = Vector{Tuple{Tuple{Int64, Int64, Int64}, Int32}}()
    num_lines::Int64 = 1
    open(path) do dataf
        for line in eachline(dataf)
            doc = parse_line(num_lines, line)
            append!(docs, doc)
            num_lines += 1
        end
    end
    return docs
end

println("loading data")
topic_assignments = load_data(data_path)

println("num_tokens = ", num_tokens)
println("num_dist_tokens = ", num_dist_tokens)
println("num_docs is ", num_docs)

topic_summary = zeros(Int64, num_topics)
topic_summary_buff = zeros(Int64, num_topics)
word_topic_table = [Dict{Int32, UInt64}() for w = 1:vocab_size]
doc_topic_table = [Dict{Int32, UInt64}() for d = 1:num_docs]
doc_topic_assignmnts = Vector{Vector{Int32}}()

srand(1)
println("initialization")
for idx in eachindex(topic_assignments)
    doc_id = topic_assignments[idx][1][1]
    word_id = topic_assignments[idx][1][2]
    word_topic_dict = word_topic_table[word_id]
    doc_topic_dict = doc_topic_table[doc_id]

    topic = rand(1:num_topics)
    topic_assignments[idx] = (topic_assignments[idx][1], topic)
    word_topic_count = get(word_topic_dict, topic, 0)
    word_topic_dict[topic] = word_topic_count + 1
    doc_topic_count = get(doc_topic_dict, topic, 0)
    doc_topic_dict[topic] = doc_topic_count + 1
    topic_summary[topic] += 1
end

println("initialization done")

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

word_topic_vec_table = map(create_word_topic_vec, word_topic_table)
word_topic_table = Vector{Dict{Int32, UInt64}}()

println("constructed word_topic_vec_table")
const beta_sum = beta * vocab_size

s_sum = 0.0
s_sum_buff = 0.0
for topic_count in topic_summary
    s_sum += alpha_beta / (beta_sum + topic_count)
end

println(s_sum)

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

q_coeff = [zeros(Float32, num_topics) for i = 1:num_docs]

for doc_id in eachindex(doc_topic_table)
    doc_topic_dict = doc_topic_table[doc_id]
    q_coeff_vec = q_coeff[doc_id]
    for topic = 1:num_topics
        count = get(doc_topic_dict, topic, 0)
        q_coeff_vec[topic] = (alpha + count) / (beta_sum + topic_summary[topic])
    end
    q_coeff[doc_id] = q_coeff_vec
end

word_log_gamma_sum = zeros(num_topics)
llh = 0.0

function find_index(vec::Vector{Int32}, key)
    for idx in eachindex(vec)
        element = vec[idx]
        if key == element
            return idx
        end
    end
    return -1
end


llh_vec = Vector{Float64}()
word_llh_vec = Vector{Float64}()

time_vec = Vector{Float64}()
start_time = now()

function sample_one_word(topic_assignment,
                         doc_topic_table,
                         word_topic_vec_table,
                         topic_summary,
                         s_sum,
                         r_sum,
                         q_coeff,
                         topic_summary_buff,
                         s_sum_buff,
                         alpha,
                         alpha_beta,
                         beta,
                         num_topics,
                         beta_sum,
                         q_term_val,
                         q_term_topic)

    doc_id = topic_assignment[1][1]
    word_id = topic_assignment[1][2]
    old_topic = topic_assignment[2]

    doc_topic_dict = doc_topic_table[doc_id]
    word_topic_vec_pair = word_topic_vec_table[word_id]
    word_topic_count_vec = word_topic_vec_pair[1]
    word_topic_vec = word_topic_vec_pair[2]
    doc_q_coeff = q_coeff[doc_id]

    doc_old_topic_count = doc_topic_dict[old_topic]

    q_sum = 0.0
    num_nonzero_q_terms = 0
    for index in eachindex(word_topic_vec)
        topic_count = word_topic_count_vec[index]
        topic = word_topic_vec[index]
        q_term = doc_q_coeff[topic] * topic_count
        num_nonzero_q_terms += 1
        q_term_val[num_nonzero_q_terms] = q_term
        q_term_topic[num_nonzero_q_terms] = topic
        q_sum += q_term
    end

    @assert q_sum >= 0
    denom = topic_summary[old_topic] + beta_sum
    s_sum_buff -= alpha_beta / denom
    s_sum_buff += alpha_beta / (denom - 1)
    r_sum[doc_id] -= (doc_old_topic_count * beta) / denom
    r_sum[doc_id] += ((doc_old_topic_count - 1) * beta) / (denom - 1)
    doc_q_coeff[old_topic] = (alpha + doc_old_topic_count - 1) / (denom - 1)
    doc_r_sum = r_sum[doc_id]
    @assert old_topic in keys(doc_topic_dict)
    doc_topic_dict[old_topic] -= 1
    if doc_topic_dict[old_topic] == 0
        delete!(doc_topic_dict, old_topic)
    end

    old_topic_index = find_index(word_topic_vec, old_topic)
    old_topic_count = word_topic_count_vec[old_topic_index]
    word_topic_count_vec[old_topic_index] = old_topic_count - 1
    q_sum -= q_term_val[old_topic_index]
    q_term_val[old_topic_index] = (old_topic_count - 1) * doc_q_coeff[old_topic]
    q_sum += q_term_val[old_topic_index]

    topic_summary_buff[old_topic] -= 1
    total_mass = q_sum + doc_r_sum + s_sum
    sample = rand() * total_mass

    new_topic::Int32 = Int32(0)
    if sample < q_sum
        for idx in 1:num_nonzero_q_terms
            sample -= q_term_val[idx]
            if sample < 0
                new_topic = q_term_topic[idx]
                break
            end
        end
        if sample >= 0
            new_topic = q_term_topic[num_nonzero_q_terms]
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
    s_sum_buff -= alpha_beta / denom
    s_sum_buff += alpha_beta / (denom + 1)
    if new_topic in keys(doc_topic_dict)
        r_sum[doc_id] -= (doc_topic_dict[new_topic] * beta) / denom
        r_sum[doc_id] += ((doc_topic_dict[new_topic] + 1) * beta) / (denom + 1)
        doc_q_coeff[new_topic] = (alpha + doc_topic_dict[new_topic] + 1) / (denom + 1)
        doc_topic_dict[new_topic] += 1
    else
        r_sum[doc_id] += beta / denom
        doc_q_coeff[new_topic] = (alpha + 1) / (denom + 1)
        doc_topic_dict[new_topic] = 1
    end
    @assert r_sum[doc_id] >= 0
    new_topic_index = find_index(word_topic_vec, new_topic)
    if new_topic_index != -1
        new_topic_count = word_topic_count_vec[new_topic_index]
        word_topic_count_vec[new_topic_index] = new_topic_count + 1
        q_sum -= q_term_val[new_topic_index]
    else
        new_topic_count = UInt64(1)
        push!(word_topic_vec, new_topic)
        push!(word_topic_count_vec, 1)
        num_nonzero_q_terms += 1
        new_topic_index = length(word_topic_vec)
    end
    q_term_val[new_topic_index] = new_topic_count * doc_q_coeff[new_topic]
    q_sum += q_term_val[new_topic_index]
    topic_summary_buff[new_topic] += 1

    return new_topic
end

function sample_all_words(topic_assignments,
                          doc_topic_table,
                          word_topic_vec_table,
                          topic_summary,
                          s_sum,
                          r_sum,
                          q_coeff,
                          topic_summary_buff,
                          s_sum_buff,
                          alpha,
                          alpha_beta,
                          beta,
                          num_topics,
                          beta_sum,
                          q_term_val,
                          q_term_topic,
                          iteration)
    for idx in eachindex(topic_assignments)
        topic_assignment = topic_assignments[idx]
        if iteration > 1
            new_topic = sample_one_word(topic_assignment,
                                        doc_topic_table,
                                        word_topic_vec_table,
                                        topic_summary,
                                        s_sum,
                                        r_sum,
                                        q_coeff,
                                        topic_summary_buff,
                                        s_sum_buff,
                                        alpha,
                                        alpha_beta,
                                        beta,
                                        num_topics,
                                        beta_sum,
                                        q_term_val,
                                        q_term_topic)
        else
            new_topic = sample_one_word(topic_assignment,
                                        doc_topic_table,
                                        word_topic_vec_table,
                                        topic_summary,
                                        s_sum,
                                        r_sum,
                                        q_coeff,
                                        topic_summary_buff,
                                        s_sum_buff,
                                        alpha,
                                        alpha_beta,
                                        beta,
                                        num_topics,
                                        beta_sum,
                                        q_term_val,
                                        q_term_topic)
        end
        topic_assignments[idx] = (topic_assignment[1], new_topic)
    end
end

function train(topic_assignments,
               doc_topic_table,
               word_topic_vec_table,
               topic_summary,
               s_sum,
               r_sum,
               q_coeff,
               topic_summary_buff,
               s_sum_buff,
               alpha,
               alpha_beta,
               beta,
               num_topics,
               beta_sum)
    last_time = start_time
    q_terms_val = Vector{Float32}(num_topics)
    q_terms_topic = Vector{Int32}(num_topics)
    for iteration = 1:num_iterations
        println("iteration = ", iteration)
        @time sample_all_words(topic_assignments,
                               doc_topic_table,
                               word_topic_vec_table,
                               topic_summary,
                               s_sum,
                               r_sum,
                               q_coeff,
                               topic_summary_buff,
                               s_sum_buff,
                               alpha,
                               alpha_beta,
                               beta,
                               num_topics,
                               beta_sum,
                               q_terms_val,
                               q_terms_topic,
                               iteration)

        topic_summary .+= topic_summary_buff
        topic_summary_buff = zeros(Int64, num_topics)
        s_sum += s_sum_buff
        s_sum_buff = 0.0

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
        llh = 0

        if iteration % 1 == 0 ||
            iteration == num_iterations
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
            word_llh = llh
            push!(word_llh_vec, word_llh)
            # compute topic likelihood
            for doc_topic_dict in doc_topic_table
                doc_log_gamma_sum = 0.0
                doc_total_word_count = 0
                for topic = 1:num_topics
                    count = get(doc_topic_dict, topic, 0)
                    doc_total_word_count += count
                    doc_log_gamma_sum += lgamma(count + alpha)
                end
                llh += doc_log_gamma_sum - lgamma(alpha * num_topics + doc_total_word_count)
                llh += lgamma(num_topics * alpha) - num_topics * lgamma(alpha)
            end
            push!(llh_vec, llh)
            curr_time = now()
            diff_time = Int(Dates.value(curr_time - last_time)) / 1000
            last_time = curr_time
            elapsed = Int(Dates.value(curr_time - start_time)) / 1000
            push!(time_vec, elapsed)
            println("iteration = ", iteration, " elapsed = ", elapsed, " llh = ", llh, " word_llh = ", word_llh)
        end
    end
end

train(topic_assignments,
      doc_topic_table,
      word_topic_vec_table,
      topic_summary,
      s_sum,
      r_sum,
      q_coeff,
      topic_summary_buff,
      s_sum_buff,
      alpha,
      alpha_beta,
      beta,
      num_topics,
      beta_sum)

println(time_vec)
println(word_llh_vec)
println(llh_vec)
