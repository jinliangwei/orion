include("utils.jl")

using Orion: load_data

const data_path = "/home/jinliang/data/ml-1m/ratings_shuffled.csv"
const K = 100
const num_iterations = 10

println("serial sgd mf starts here!")
ratings = load_data(data_path)
