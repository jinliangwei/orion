include("/users/jinlianw/orion.git/src/julia/orion.jl")

# set path to the C++ runtime library
Orion.set_lib_path("/users/jinlianw/orion.git/lib/liborion_driver.so")
# test library path
Orion.helloworld()

const master_ip = "10.117.1.14"
#const master_ip = "127.0.0.1"
const master_port = 10000
const comm_buff_capacity = 1024
const num_executors = 1
const num_servers = 1

Orion.glog_init()
Orion.init(master_ip, master_port, comm_buff_capacity, num_executors, num_servers)

const data_path = "file:///proj/BigLearning/jinlianw/data/nytimes.dat.perm"
#const data_path = "file:///proj/BigLearning/jinlianw/data/nytimes.dat"
#const data_path = "file:///proj/BigLearning/jinlianw/data/nytimes.dat.perm.4"
#const data_path = "file:///proj/BigLearning/jinlianw/data/pubmed.dat"
#const data_path = "file:///proj/BigLearning/jinlianw/data/clueweb.libsvm.8M"
Orion.@share const num_topics = 100

Orion.@share const alpha = Float32(0.1)
Orion.@share const beta = Float32(0.1)
const num_iterations = 2

const alpha_beta = alpha * beta

Orion.@share num_tokens = 0
Orion.@share num_dist_tokens = 0

Orion.@share function parse_line(line_num::Int64, line::AbstractString)::Vector{Tuple{Tuple{Int64, Int64, Int64},
                                                                                      Int32}
                                                                                }
    # Parse each line to be word counts
    tokens = split(strip(line),  ' ')
    token_vec = Vector{Tuple{Tuple{Int64, Int64, Int64},
                            Int32}
                      }()
    idx = 1
    for token in tokens[2:end]
        word_count = split(token, ":")
        word_id = parse(Int64, word_count[1]) + 1
        count = parse(Int32, word_count[2])
        global num_tokens += count
        global num_dist_tokens += 1
        for c = 1:count
            push!(token_vec, ((c, word_id, line_num), 0))
        end
        idx += 1
    end
    return token_vec
end

Orion.@dist_array topic_assignments = Orion.text_file(data_path, parse_line,
                                                      is_dense = false,
                                                      with_line_number = true,
                                                      new_keys = true,
                                                      flatten_results = true,
                                                      num_dims = 3)

Orion.materialize(topic_assignments)

(max_token_per_word, vocab_size, num_docs) = size(topic_assignments)
num_tokens = Orion.get_aggregated_value(:num_tokens, :+)
println("num_tokens = ", num_tokens)
num_dist_tokens = Orion.get_aggregated_value(:num_dist_tokens, :+)
println("num_dist_tokens = ", num_dist_tokens)
println("vocab_size = ", vocab_size)
println("num_docs is ", num_docs)

docs_histogram = Orion.compute_histogram(topic_assignments, 2, 64)
println("before shuffle")
println(sort!(map(x -> (x[1], Int64(x[2])), docs_histogram), by = x->x[2])
        )

docs_histogram = Orion.compute_histogram(topic_assignments, 3, 64)
println("before shuffle")
println(sort!(map(x -> (x[1], Int64(x[2])), docs_histogram), by = x->x[2])
        )

@time Orion.random_remap_keys!(topic_assignments, (2,))
@time Orion.random_remap_keys!(topic_assignments, (3,))

docs_histogram = Orion.compute_histogram(topic_assignments, 2, 64)
println("after shuffle")
println(sort!(map(x -> (x[1], Int64(x[2])), docs_histogram), by = x->x[2])
        )

docs_histogram = Orion.compute_histogram(topic_assignments, 3, 64)
println("after shuffle")
println(sort!(map(x -> (x[1], Int64(x[2])), docs_histogram), by = x->x[2])
        )

Orion.@dist_array topic_summary = Orion.fill(zeros(Int64, num_topics), 1)
Orion.materialize(topic_summary)

Orion.@dist_array word_topic_table = Orion.fill(zeros(UInt64, num_topics), vocab_size)
Orion.materialize(word_topic_table)

Orion.@dist_array doc_topic_table = Orion.fill(Dict{Int32, UInt64}(), num_docs)
Orion.materialize(doc_topic_table)

Orion.@dist_array topic_summary_buff = Orion.create_dense_dist_array_buffer((1,),
                                                                            zeros(UInt64, num_topics))
Orion.materialize(topic_summary_buff)

Orion.@share function apply_buffered_vec(vec_key,
                                         value_vec::Vector,
                                         update_vec::Vector)
    for idx in eachindex(update_vec)
        value_vec[idx] += update_vec[idx]
    end
    return value_vec
end

Orion.set_write_buffer(topic_summary_buff, topic_summary, apply_buffered_vec,
                       max_delay = 6000000000)

Orion.@share srand(1)

println("initialization")
Orion.@parallel_for for topic_assignment_pair in topic_assignments
    doc_id = topic_assignment_pair[1][3]
    word_id = topic_assignment_pair[1][2]
    word_topic_vec = word_topic_table[word_id]
    doc_topic_dict = doc_topic_table[doc_id]
    topic_summary_buff_vec = topic_summary_buff[1]

    topic = rand(1:num_topics)
    word_topic_vec[topic] += 1
    doc_topic_count = get(doc_topic_dict, topic, 0)
    doc_topic_dict[topic] = doc_topic_count + 1
    topic_summary_buff_vec[topic] += 1
    word_topic_table[word_id] = word_topic_vec
    doc_topic_table[doc_id] = doc_topic_dict
    topic_summary_buff[1] = topic_summary_buff_vec
    topic_assignment_pair = (topic_assignment_pair[1], topic)
end

Orion.@share function create_word_topic_vec(word_topic_dict::Vector{UInt64}
                                            )::Tuple{Vector{UInt64}, Vector{Int32}}
    num_nonzero_counts = 0
    for count in word_topic_dict
        if count > 0
            num_nonzero_counts += 1
        end
    end
    word_topic_count_vec = Vector{UInt64}(num_nonzero_counts)
    word_topic_index_vec = Vector{Tuple{Int32, Int32}}(num_nonzero_counts)
    index = 1
    for topic in eachindex(word_topic_dict)
        topic_count = word_topic_dict[topic]
        if topic_count == 0
            continue
        end
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

Orion.@dist_array word_topic_vec_table = Orion.map(word_topic_table,
                                                   create_word_topic_vec,
                                                   map_values = true)
Orion.materialize(word_topic_vec_table)
Orion.delete_dist_array(word_topic_table)

println("initializing s_sum")

Orion.@accumulator s_sum_accum = Float32(0.0)

const beta_sum = Float32(beta * vocab_size)

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

Orion.@dist_array s_sum_buff = Orion.create_dense_dist_array_buffer((s_sum.dims...), Float32(0.0))
Orion.materialize(s_sum_buff)

Orion.set_write_buffer(s_sum_buff, s_sum, apply_buffered_update, max_delay = 6000000000)
println("initializing r_sum")
Orion.@dist_array r_sum = Orion.fill(Float32(0.0), num_docs)
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
Orion.@dist_array q_coeff = Orion.fill(zeros(Float32, num_topics), num_docs)
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

Orion.@dist_array word_log_gamma_sum = Orion.fill(0.0, (num_topics,))
Orion.materialize(word_log_gamma_sum)

Orion.@dist_array word_log_gamma_sum_buff = Orion.create_dense_dist_array_buffer((num_topics,), 0.0)
Orion.materialize(word_log_gamma_sum_buff)

Orion.set_write_buffer(word_log_gamma_sum_buff, word_log_gamma_sum, apply_buffered_update)

Orion.@accumulator llh = Float32(0.0)

Orion.@share function find_index(vec::Vector{Int32}, key)
    for idx in eachindex(vec)
        element = vec[idx]
        if key == element
            return idx
        end
    end
    return -1
end

llh_vec = Vector{Float32}()
word_llh_vec = Vector{Float32}()

time_vec = Vector{Float64}()
start_time = now()
last_time = start_time

Orion.@accumulator old_word_id = Int32(0)
Orion.@accumulator old_doc_id = Int32(0)
Orion.@accumulator q_sum = Float32(0.0)
Orion.@accumulator num_nonzero_q_terms = UInt64(0)

Orion.@share const q_term_val = Vector{Float32}(num_topics)
Orion.@share const q_term_topic = Vector{Int32}(num_topics)

@time for iteration = 1:num_iterations
    println("iteration = ", iteration)
    Orion.@parallel_for repeated histogram_partitioned for topic_assignment_pair in topic_assignments
        doc_id = topic_assignment_pair[1][3]
        word_id = topic_assignment_pair[1][2]
        old_topic = topic_assignment_pair[2]

        doc_topic_dict = doc_topic_table[doc_id]
        word_topic_vec_pair = word_topic_vec_table[word_id]
        word_topic_count_vec = word_topic_vec_pair[1]
        word_topic_vec = word_topic_vec_pair[2]
        topic_summary_vec = topic_summary[1]
        doc_q_coeff = q_coeff[doc_id]
        s_sum_val = s_sum[1]

        if old_word_id != word_id ||
            old_doc_id != doc_id
            q_sum = Float32(0.0)
            num_nonzero_q_terms = UInt64(0)
            for index in eachindex(word_topic_vec)
                topic_count = word_topic_count_vec[index]
                topic = word_topic_vec[index]
                q_term = doc_q_coeff[topic] * topic_count
                num_nonzero_q_terms += 1
                q_term_val[num_nonzero_q_terms] = q_term
                q_term_topic[num_nonzero_q_terms] = topic
                q_sum += q_term
            end
            old_word_id = word_id
            old_doc_id = doc_id
        end

        denom = topic_summary_vec[old_topic] + beta_sum
        s_sum_buff[1] -= alpha_beta / denom
        s_sum_buff[1] += alpha_beta / (denom - 1)
        r_sum[doc_id] -= (doc_topic_dict[old_topic] * beta) / denom
        r_sum[doc_id] += ((doc_topic_dict[old_topic] - 1) * beta) / (denom - 1)
        doc_q_coeff[old_topic] = (alpha + doc_topic_dict[old_topic] - 1) / (denom - 1)
        doc_r_sum = r_sum[doc_id]
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
        topic_summary_buff[1][old_topic] -= 1

        total_mass = q_sum + doc_r_sum + s_sum_val
        sample = rand() * total_mass
        new_topic = Int32(0)
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
        elseif sample <= q_sum + doc_r_sum + s_sum_val
            sample -= q_sum + doc_r_sum
            sample /= alpha_beta
            for t in 1:num_topics
                sample -= 1.0 / (topic_summary_vec[t] + beta_sum)
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

        denom = topic_summary_vec[new_topic] + beta_sum
        s_sum_buff[1] -= alpha_beta / denom
        s_sum_buff[1] += alpha_beta / (denom + 1)
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
            q_sum -= q_term_val[new_topic_index]
        else
            push!(word_topic_vec, new_topic)
            push!(word_topic_count_vec, 1)
            new_topic_count = UInt64(1)
            new_topic_index = length(word_topic_vec)
            num_nonzero_q_terms += 1
        end
        q_term_val[new_topic_index] = new_topic_count * doc_q_coeff[new_topic]
        q_term_topic[new_topic_index] = new_topic
        q_sum += q_term_val[new_topic_index]

        topic_summary_buff[1][new_topic] += 1

        doc_topic_table[doc_id] = doc_topic_dict
        word_topic_vec_table[word_id] = (word_topic_count_vec, word_topic_vec)
        q_coeff[doc_id] = doc_q_coeff
        topic_assignment_pair = (topic_assignment_pair[1], new_topic)
    end

    Orion.@parallel_for for word_topic_vec_pair in word_topic_vec_table
        word_topic_count_vec = word_topic_vec_pair[2][1]
        word_topic_vec = word_topic_vec_pair[2][2]
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
        temp_word_topic_count_vec = word_topic_count_vec
        word_topic_count_vec = Vector{Int64}(length(temp_word_topic_count_vec))
        word_topic_count_vec .= temp_word_topic_count_vec

        resize!(word_topic_vec, num_nonzeros)
        temp_word_topic_vec = word_topic_vec
        word_topic_vec = Vector{Int32}(length(temp_word_topic_vec))
        word_topic_vec .= temp_word_topic_vec

        @assert length(word_topic_count_vec) == length(word_topic_vec)
        word_topic_vec_pair = (word_topic_vec_pair[1], (word_topic_count_vec, word_topic_vec))
    end

    if iteration % 1 == 0 ||
        iteration == num_iterations
        Orion.@parallel_for for word_topic_pair in word_topic_vec_table
            word_topic_count_vec = word_topic_pair[2][1]
            word_topic_vec = word_topic_pair[2][2]
            @assert length(word_topic_count_vec) == length(word_topic_vec)
            for idx in eachindex(word_topic_vec)
                topic = word_topic_vec[idx]
                count = word_topic_count_vec[idx]
                word_log_gamma_sum_buff[topic] += lgamma(count + beta)
            end
            llh += (num_topics - length(word_topic_vec)) * lgamma(beta)
        end

        Orion.@parallel_for for topic_log_gamma_sum in word_log_gamma_sum
            topic = topic_log_gamma_sum[1][1]
            topic_log_gamma_sum_val = topic_log_gamma_sum[2]
            topic_summary_vec = topic_summary[1]
            llh += topic_log_gamma_sum_val - lgamma(vocab_size * beta + topic_summary_vec[topic])
            llh += lgamma(vocab_size * beta) - vocab_size * lgamma(beta)
            topic_log_gamma_sum = ((topic,), 0.0)
        end
        word_llh = Orion.get_aggregated_value(:llh, :+)
        push!(word_llh_vec, word_llh)
        # compute topic likelihood
        Orion.@parallel_for for doc_topic_dict_pair in doc_topic_table
            doc_topic_dict = doc_topic_dict_pair[2]
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
        llh = Orion.get_aggregated_value(:llh, :+)
        Orion.reset_accumulator(:llh)
        push!(llh_vec, llh)
        curr_time = now()
        diff_time = Int(Dates.value(curr_time - last_time)) / 1000
        last_time = curr_time
        elapsed = Int(Dates.value(curr_time - start_time)) / 1000
        push!(time_vec, elapsed)
        println("iteration = ", iteration, " elapsed = ", elapsed, " iter_time = ", diff_time,
                " llh = ", llh, " word_llh = ", word_llh)
    end
end

println(time_vec)
println(word_llh_vec)
println(llh_vec)
Orion.stop()

llh_fobj = open("results/" * split(PROGRAM_FILE, "/")[end] * "-" *
                 split(data_path, "/")[end] * "-" * string(num_executors) * "-" *
                 string(num_topics) * "-" * string(now()) * ".llh", "w")
for idx in eachindex(time_vec)
    write(llh_fobj, string(idx) * "\t" * string(time_vec[idx]) * "\t" * string(llh_vec[idx]) * "\n")
end
close(llh_fobj)
exit()
