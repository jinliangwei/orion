include("/users/jinlianw/orion.git/src/julia/orion.jl")

# set path to the C++ runtime library
Orion.set_lib_path("/users/jinlianw/orion.git/lib/liborion_driver.so")
# test library path
Orion.helloworld()

const master_ip = "127.0.0.1"
const master_port = 10000
const comm_buff_capacity = 1024
const num_executors = 1
const num_servers = 1

Orion.glog_init()
Orion.init(master_ip, master_port, comm_buff_capacity, num_executors, num_servers)

const data_path = "file:///proj/BigLearning/jinlianw/data/20news.dat.200"
const num_topics = 100

Orion.@share const alpha = 0.1
Orion.@share const beta = 0.1
const num_iterations = 10

const alpha_beta = alpha * beta

Orion.@share function parse_line(line_num::Int64, line::AbstractString)::Vector{Tuple{Tuple{Int64, Int32},
                                                                                   Int64}
                                                                                }
    # Parse each line to be word counts
    tokens = split(strip(line),  ' ')
    pair_vec = Vector{Tuple{Tuple{Int64, Int32},
                         Int64
                         }
                   }()
    for token in tokens[2:end]
        word_count = split(token, ":")
        word_id = parse(Int32, word_count[1]) + 1
        word_count = parse(Int64, word_count[2])
        push!(pair_vec, ((line_num, word_id), word_count))
    end
    return pair_vec
end

Orion.@dist_array doc_word_counts = Orion.text_file(data_path, parse_line,
                                                    is_dense = false,
                                                    with_line_number = true,
                                                    new_keys = true,
                                                    flatten_results = true,
                                                    num_dims = 2)

Orion.materialize(doc_word_counts)

(num_docs, vocab_size) = size(doc_word_counts)
println("vocab_size = ", vocab_size)
println("num_docs is ", num_docs)

Orion.@share function init_topic_assignments(word_count::Int64)::Vector{Int32}
    return Vector{Int32}(word_count)
end

Orion.@dist_array topic_assignments = Orion.map(doc_word_counts,
                                                init_topic_assignments,
                                                map_values = true)
Orion.materialize(topic_assignments)

Orion.@dist_array topic_summary = Orion.fill(zeros(Int64, num_topics), 1)
Orion.materialize(topic_summary)

Orion.@dist_array word_topic_table = Orion.fill(Dict{Int32, Int64}(), vocab_size)
Orion.materialize(word_topic_table)

Orion.@dist_array doc_topic_table = Orion.fill(Dict{Int32, Int64}(), num_docs)
Orion.materialize(doc_topic_table)

Orion.@dist_array topic_summary_buff = Orion.create_dense_dist_array_buffer((1,),
                                                                            zeros(Int64, num_topics))
Orion.materialize(topic_summary_buff)

Orion.@share function apply_buffered_dict(dict_key,
                                          value_dict::Dict{Int32, Int64},
                                          update_dict::Dict{Int32, Int64})
    for (key, val) in value_dict
        if key in keys(value_dict)
	    value_dict[key] += val
            if value_dict[key] == 0
                delete!(value_dict, key)
            end
        else
            value_dict[key] = value
        end
    end
    return value_dict
end

Orion.@share function apply_buffered_vec(vec_key,
                                         value_vec::Vector,
                                         update_vec::Vector)
    for idx in eachindex(update_vec)
        value_vec[idx] += update_vec[idx]
    end
    return value_vec
end

Orion.set_write_buffer(topic_summary_buff, topic_summary, apply_buffered_vec)

println("initialization")
Orion.@parallel_for for doc_word_count in doc_word_counts
    doc_id = doc_word_count[1][1]
    word_id = doc_word_count[1][2]
    count = doc_word_count[2]
    assignment_vec = topic_assignments[doc_id, word_id]
    word_topic_dict = word_topic_table[word_id]
    doc_topic_dict = doc_topic_table[doc_id]
    topic_summary_buff_vec = topic_summary_buff[1]

    for c = 1:count
        topic = rand(1:num_topics)
        assignment_vec[c] = topic
        word_topic_count = get(word_topic_dict, topic, 0)
        word_topic_dict[topic] = word_topic_count + 1
        doc_topic_count = get(doc_topic_dict, topic, 0)
        doc_topic_dict[topic] = doc_topic_count + 1
        topic_summary_buff_vec[topic] += 1
    end
    word_topic_table[word_id] = word_topic_dict
    doc_topic_table[doc_id] = doc_topic_dict
    topic_summary_buff[1] = topic_summary_buff_vec
end

println("initializing s_sum")

Orion.@accumulator s_sum_accum = 0

const beta_sum = beta * vocab_size

Orion.@parallel_for for topic_summary_pair in topic_summary
    topic_summary_vec = topic_summary_pair[2]
    for topic_count in topic_summary_vec
        s_sum_accum += alpha_beta / (beta_sum + topic_count)
    end
end

s_sum_accum = Orion.get_aggregated_value(:s_sum_accum, :+)
println("s_sum_accum = ", s_sum_accum)

Orion.@dist_array s_sum = Orion.fill(s_sum_accum, 1)
Orion.materialize(s_sum)

Orion.@share function apply_buffered_update(key::Int64, value, update)
    return value + update
end

Orion.@dist_array s_sum_buff = Orion.create_dense_dist_array_buffer((s_sum.dims...), 0.0)
Orion.materialize(s_sum_buff)
Orion.set_write_buffer(s_sum_buff, s_sum, apply_buffered_update)

println("initializing r_sum")
Orion.@dist_array r_sum = Orion.fill(0.0, num_docs)
Orion.materialize(r_sum)

Orion.@parallel_for for doc_topic_dict_pair in doc_topic_table
    doc_id = doc_topic_dict_pair[1][1]
    doc_topic_dict = doc_topic_dict_pair[2]
    topic_summary_vec = topic_summary[1]
    doc_r_sum = 0.0
    for topic_count in doc_topic_dict
        topic = topic_count[1]
        count = topic_count[2]
        doc_r_sum += count * beta / (beta_sum + topic_summary_vec[topic])
    end
    r_sum[doc_id] = doc_r_sum
end
println("initializing q_coeff")
Orion.@dist_array q_coeff = Orion.fill(zeros(Float64, num_topics), num_docs)
Orion.materialize(q_coeff)

Orion.@parallel_for for doc_topic_dict_tuple in doc_topic_table
    doc_id = doc_topic_dict_tuple[1][1]
    doc_topic_dict = doc_topic_dict_tuple[2]
    topic_summary_vec = topic_summary[1]
    q_coeff_vec = q_coeff[doc_id]
    for topic = 1:num_topics
        count = get(doc_topic_dict, topic, 0)
        q_coeff_vec[topic] = (alpha + count) / (beta_sum + topic_summary_vec[topic])
    end
    q_coeff[doc_id] = q_coeff_vec
end

llh_vec = Vector{Float64}()
q_terms = Vector{Tuple{Float32, Int32}}(num_topics)

Orion.@dist_array log_gamma_sum = Orion.fill([0.0, 0.0], (num_topics,))
Orion.materialize(log_gamma_sum)

Orion.@dist_array log_gamma_sum_buff = Orion.create_dense_dist_array_buffer((num_topics,), [0.0, 0.0])
Orion.materialize(log_gamma_sum_buff)

Orion.set_write_buffer(log_gamma_sum_buff, log_gamma_sum, apply_buffered_vec)

Orion.@accumulator llh = 0.0

@time for iteration = 1:num_iterations
    println("iteration = ", iteration)
    Orion.@parallel_for for doc_word_count in doc_word_counts
        doc_id = doc_word_count[1][1]
        word_id = doc_word_count[1][2]
        word_count = doc_word_count[2]
        assignment_vec = topic_assignments[doc_id, word_id]
        doc_topic_dict = doc_topic_table[doc_id]
        word_topic_dict = word_topic_table[word_id]
        topic_summary_vec = topic_summary[1]
        doc_q_coeff = q_coeff[doc_id]
        s_sum_val = s_sum[1]
        for c = 1:word_count
            println("c = ", c)
            old_topic = assignment_vec[c]
            denom = topic_summary_vec[old_topic] + beta_sum
            println("denom = ", denom)
            println("alpha_beta = ", alpha_beta)
            println("alpha_beta / denom = ", alpha_beta / denom)
            s_sum_buff[1] -= alpha_beta / denom
            s_sum_buff[1] += alpha_beta / (denom - 1)
            println("s_sum_buff[1] = ", s_sum_buff[1])
            r_sum[doc_id] -= (doc_topic_dict[old_topic] * beta) / denom
            r_sum[doc_id] += ((doc_topic_dict[old_topic] - 1) * beta) / (denom - 1)
            println("r_sum[doc_id] = ", r_sum[doc_id])
            doc_q_coeff[old_topic] = (alpha + doc_topic_dict[old_topic] - 1) / (denom - 1)
            println("doc_q_coeff[old_topic] = ", doc_q_coeff[old_topic])
            doc_r_sum = r_sum[doc_id]
            println("doc_r_sum = ", doc_r_sum)
            doc_topic_dict[old_topic] -= 1
            if doc_topic_dict[old_topic] == 0
                delete!(doc_topic_dict, old_topic)
            end

            word_topic_dict[old_topic] -= 1
            if word_topic_dict[old_topic] == 0
                delete!(word_topic_dict, old_topic)
            end
            topic_summary_buff[1][old_topic] -= 1

            q_sum = 0.0
            index = 1
            for (topic, topic_count) in word_topic_dict
                q_term = (topic == old_topic) ? (doc_q_coeff[topic] * topic_count - 1) :
                    doc_q_coeff[topic] * topic_count
                q_terms[index] = (q_term, topic)
                q_sum += q_term
                index += 1
            end
            total_mass = q_sum + doc_r_sum + s_sum_val
            println("total_mass = ", total_mass)
            sample = rand() * total_mass
            println("sample = ", sample, " q_sum = ", q_sum,
                    " q_sum + doc_r_sum = ", q_sum + doc_r_sum,
                    " q_sum + doc_r_sum + s_sum_val = ", q_sum + doc_r_sum + s_sum_val)
            new_topic = 0
            if sample < q_sum
                for q_term in q_terms
                    sample -= q_term[1]
                    if sample < 0
                        new_topic = q_term[2]
                        break
                    end
                end
                if sample > 0
                    @assert sample < 0 sample
                    new_topic = q_terms[end][2]
                end
            elseif sample < q_sum + doc_r_sum
                sample -= q_sum
                sample /= beta
                topic_last = 1
                for (topic, topic_count) in doc_topic_dict
                    sample -= topic_count / (topic_summary_vec[topic] + beta_sum)
                    topic_last = topic
                    if sample < 0
                        new_topic = topic
                        break
                    end
                end
                if sample >= 0
                    new_topic = topic_last
                end
            elseif sample < q_sum + doc_r_sum + s_sum_val
                sample -= q_sum + doc_r_sum
                sample /= alpha_beta
                for t in 1:num_topics
                    sample -= 1.0 / (topic_summary_vec[t] + beta_sum)
                    if sample < 0
                        new_topic = t
                        break
                    end
                end
                if sample >= 0
                    new_topic = num_topics
                end
            else
                error("sample = ", sample, " total_mass = ", total_mass)
            end
            println("new_topic = ", new_topic)
            denom = topic_summary_vec[new_topic] + beta_sum
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
            if new_topic in keys(word_topic_dict)
                word_topic_dict[new_topic] += 1
            else
                word_topic_dict[new_topic] = 1
            end

            topic_summary_buff[1][new_topic] += 1
            assignment_vec[c] = new_topic
        end

        topic_assignments[doc_id, word_id] = assignment_vec
        doc_topic_table[doc_id] = doc_topic_dict
        word_topic_dict = word_topic_table[word_id] = word_topic_dict
        q_coeff[doc_id] = doc_q_coeff
    end

    if iteration % 1 == 0 ||
        iteration == num_iterations
        Orion.@parallel_for for word_topic_dict_pair in word_topic_table
            word_id = word_topic_dict_pair[1][1]
            word_topic_dict = word_topic_dict_pair[2]
            for topic = 1:num_topics
                count = get(word_topic_dict, topic, 0)
                value = count + beta
                log_gamma_sum_buff[topic][1] += lgamma(value)
                log_gamma_sum_buff[topic][2] += value
            end
        end

        Orion.@parallel_for for topic_log_gamma_sum in log_gamma_sum
            log_gamma_sum_vals = topic_log_gamma_sum[2]
            llh += log_gamma_sum_vals[1] - lgamma(log_gamma_sum_vals[2])
            llh -= vocab_size * lgamma(beta) - lgamma(vocab_size * beta)
            topic_log_gamma_sum[2][1] = 0.0
            topic_log_gamma_sum[2][2] = 0.0
        end

        Orion.@parallel_for for doc_topic_dict_pair in doc_topic_table
            doc_topic_dict = doc_topic_dict_pair[2]
            log_gamma_sum = 0.0
            value_sum = 0.0
            for topic = 1:num_topics
                count = get(doc_topic_dict, topic, 0)
                value = count + beta
                log_gamma_sum += lgamma(value)
                value_sum += value
            end
            llh += log_gamma_sum - lgamma(value_sum)
            llh -= num_topics * lgamma(alpha) - lgamma(num_topics * alpha)
        end
    end
    llh = Orion.get_aggregated_value(:llh, :+)
    push!(llh_vec, llh)
    println("iteration = ", iteration, " llh = ", llh)
    Orion.reset_accumulator(:llh)
end

println(llh_vec)
Orion.stop()
exit()
