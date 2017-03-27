push!(LOAD_PATH, "/home/ubuntu/orion/src/julia/")
import orion
import Orion

Orion.set_lib_path("/home/ubuntu/orion/lib/liborion.so")
Orion.helloworld()
Orion.glog_init(C_NULL)
Orion.init("127.0.0.1", 10000, 1024)
ret = Orion.execute_code(0, "sqrt(2.0)", Float64)
println(ret)
Orion.stop()

ret = Orion.get_result_type_value(Float64)
println(ret)


#ccall((:orion_helloworld, "/home/ubuntu/orion/lib/liborion.so"), Void, ())


#Orion.create_driver("127.0.0.1", 10000, 1024)
