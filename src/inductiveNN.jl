using DataFrames
using Flux
using MLUtils
using Random
using ProgressMeter

GLMakie.activate!()
# X = skin_arr_resampled[empty_filter, :]'
# y = position_arr_resampled[empty_filter, [1, 3]]'


positions_arr
iposition_arr

X = iposition_arr ./ 1.0e5
y = positions_arr[[1, 2], :] ./ 1.0e5
train_inds, test_inds = splitobs(size(X, 2), at=0.9)
inds = shuffle(1:size(X, 2))
train_loader = Flux.DataLoader((X[:, inds[train_inds]], y[:, inds[train_inds]]) |> gpu, batchsize=32, shuffle=true)
val_loader = Flux.DataLoader((X[:, test_inds], y[:, test_inds]) |> gpu, batchsize=length(test_inds))

model = Chain(
    Dense(2, 128, relu),
    Dense(128, 128, relu),
    Dense(128, 128, relu),
    Dense(128, 2),
) |> gpu

function criterion(result, label)
    loss = Flux.mse(result, label)
    return loss
end

opt_state = Flux.setup(Adam(0.001), model)
epochs = 1000
f = Figure()
ax1 = Makie.Axis(f[1, 1], yscale=log10)
ax2 = Makie.Axis(f[1, 2])

train_losses = Vector{Float32}()
val_losses = Vector{Float32}()
@showprogress for epoch in 1:epochs
    loss_ = 0
    loss_cnt = 0
    for (i, data) in enumerate(train_loader)
        input, label = data

        val, grads = Flux.withgradient(model) do m
            result = m(input)
            loss = criterion(result, label)
        end
        loss_ += val * size(input, 2)
        loss_cnt += size(input, 2)

        # Detect loss of Inf or NaN. Print a warning, and then skip update!
        if !isfinite(val)
            @warn "loss is $val on item $i" epoch
            continue
        end

        Flux.update!(opt_state, model, grads[1])
    end
    push!(train_losses, loss_ / loss_cnt)

    # Validation loss
    input, label = first(val_loader)
    result = model(input)
    loss = criterion(result, label)
    push!(val_losses, loss)

    # clear display
    # IJulia.clear_output(true)
    println("Epoch: $epoch, Train loss: $(train_losses[end]), Val loss: $(val_losses[end])")

    # plot loss
    empty!(ax1)
    lines!(ax1, 1:epoch, train_losses, color=:blue, label="Train loss", linewidth=2)
    lines!(ax1, 1:epoch, val_losses, color=:red, label="Val loss", linewidth=2)
    autolimits!(ax1)
    axislegend(ax1, merge=true, unique=true)

    # # plot prediction
    # empty!(ax2)
    # input, label = first(train_loader)
    # result = model(input)
    # input = input |> cpu
    # label = label |> cpu
    # result = result |> cpu
    # scatter!(ax2, result[1, :], result[2, :], color=:red, label="Predicted")
    # scatter!(ax2, label[1, :], label[2, :], color=:blue, label="Ground truth")
    # linesegments!(ax2, [result[1, :] label[1, :]]'[:], [result[2, :] label[2, :]]'[:], color=:black)
    # axislegend(ax2, merge=true, unique=true)
    # limits!(ax2, 0.13, 0.5, -0.1, 0.1)
    display(f)

end

# predict full data
all_data = Flux.DataLoader((X, y) |> gpu, batchsize=length(y))
# result = model(first(all_data)[1])
result = model(first(val_loader)[1])


# plot prediction
f = Figure()
ax = Makie.Axis(f[1, 1])
scatterlines!(ax, result[1, :], result[2, :], color=:red, label="Predicted")
# scatterlines!(ax, y[1, :], y[2, :], color=:blue, label="Ground truth")
display(f)