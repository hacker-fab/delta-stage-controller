# using Pkg
# ENV["PYTHON"] = abspath("venv/bin/python")
# Pkg.build("PyCall")
using StatsBase
using ZMQ
using ZeroMQ_jll
using MsgPack
using Statistics
using PyCall
using GLMakie
using LinearAlgebra
using ProgressBars
using Serialization
using DataFrames
using Flux
pushfirst!(pyimport("sys")."path", "src")
GLMakie.activate!(inline=false)


py"""
from stage.sanga import SangaStage, SangaDeltaStage
ss = SangaDeltaStage(port = "/dev/ttyACM0")
"""


window_size = 5000
is = Observable(zeros(window_size, 2))
xs = @lift($is[:, 1])
ys = @lift($is[:, 2])

f = Figure()
ax = Axis(f[1, 1])
lines!(ax, 1:window_size, xs, color = :blue)
lines!(ax, 1:window_size, ys, color = :red)
display(f)

py"""ss.move_rel_delta([-3000, -3000, -3000])"""
py"""ss.move_rel_delta([200, 200, 200])"""
curr_position = py"""ss.position"""
x_sweep = curr_position[1]:5:curr_position[1] + 400
y_sweep = curr_position[2]:5:curr_position[2] + 400

# traverse in z shape, reversing order in y
poses = []
for y in y_sweep
    for x in x_sweep
        push!(poses, [x, y, curr_position[3]])
    end
    x_sweep = reverse(x_sweep)
end
poses = hcat(poses...)
diffs = diff(poses, dims = 2)

inductances_x = []
inductances_y = []
positions = []
for pose in eachcol(poses)
    # move
    py"""ss.move_abs($pose)"""

    is[] = circshift(is[], (1, 0))
    is[][1, 1] = parse(Int, py"""ss.board.query("i0?")""")
    is[][1, 2] = parse(Int, py"""ss.board.query("i1?")""")
    position = py"""ss.position"""
    push!(inductances_x, is[][1, 1])
    push!(inductances_y, is[][1, 2])
    push!(positions, position)

    notify(is)
    println(position)
end

curr_position = py"""ss.position"""
py"""ss.move_abs([88510, 4243, -76239])"""

while true
    is[] = circshift(is[], (1, 0))
    is[][1, 1] = parse(Int, py"""ss.board.query("i0?")""")
    is[][1, 2] = parse(Int, py"""ss.board.query("i1?")""")
    notify(is)
    println(is[][1, 1])
end




# # %% start
# from stage.sanga import SangaStage, SangaDeltaStage
# ss = SangaDeltaStage(port = "/dev/ttyACM0")

# # %% move
# ss.move_rel(-200, axis = "z")

# # %%
# ss.board.query("i?")
# # %%

# num = 0
# import time
# start_t  = time.time()
# for i in range(100):
#     ss.board.query("i?")
#     ss.move_rel(0, axis = "z")
#     num += 1
# end_t = time.time()

# fps = num / (end_t - start_t)
# print(fps)
# # %%
