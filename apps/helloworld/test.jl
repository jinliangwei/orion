include("/users/jinlianw/orion.git/src/julia/orion.jl")

Orion.set_lib_path("/users/jinlianw/orion.git/lib/liborion_driver.so")
Orion.helloworld()

const master_ip = "127.0.0.1"
#const master_ip = "10.117.1.14"
const master_port = 10000
const comm_buff_capacity = 1024
const num_executors = 4

Orion.glog_init()
Orion.init(master_ip, master_port, comm_buff_capacity, num_executors)

@Orion.accumulator cnt = 0 +
println("cnt = ", cnt)

cnt = Orion.get_accumulator_value(:cnt)

println("cnt = ", cnt)

Orion.stop()
