include("/home/ubuntu/orion/src/julia/orion.jl")
println("application started")

# set path to the C++ runtime library
Orion.set_lib_path("/home/ubuntu/orion/lib/liborion_driver.so")
# test library path
Orion.helloworld()

const master_ip = "127.0.0.1"
const master_port = 10000
const comm_buff_capacity = 1024
const num_executors = 2

# initialize logging of the runtime library
Orion.glog_init()
Orion.init(master_ip, master_port, comm_buff_capacity, num_executors)

const K = 100
Orion.define_var(:K)
Orion.eval_expr_on_all(:(Array{Int32,1}()), :OrionGen)

Orion.stop()
