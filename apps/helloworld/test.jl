include("/home/ubuntu/orion/src/julia/orion_worker.jl")

OrionWorker.set_lib_path("/home/ubuntu/orion/lib/liborion.so")
#OrionWorker.helloworld()
OrionWorker.load_constants()
ret = OrionWorker.dist_array_read(Ptr{Void}(0), (1,))
println(ret)

ret = OrionWorker.dist_array_read(Ptr{Void}(0), (2:4, 2:4))
println(ret)
