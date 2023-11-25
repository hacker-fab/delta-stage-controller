using LibSerialPort
using GLMakie
using LinearAlgebra
using ProgressMeter
using Serialization
using Distributions
using LowLevelParticleFilters
using LinearAlgebra
using ForwardDiff
using Flux
using ControlSystemsBase
using LsqFit
GLMakie.activate!(inline=false)


# Simulation
bootstrap_stepper_u = function (t)
    if t >= 0.0 && t < 1.0
        return [sin((t - 0) * (5 * 2 * pi)), 0.0, 0.0]
    elseif t >= 1.0 && t < 2.0
        return [0.0, sin((t - 1) * (5 * 2 * pi)), 0.0]
    elseif t >= 2.0 && t < 3.0
        return [0.0, 0.0, 10 * sin((t - 2) * (5 * 2 * pi))]
    else
        return [sin((t - 0) * (5 * 2 * pi)), cos((t - 0) * (5 * 2 * pi)), sin((t - 0) * (5 * 2 * pi))]
    end
end
# bootstrap_stepper_u = function (t)
#     return [sin((t - 0) * (5 * 2 * pi)), cos((t - 0) * (5 * 2 * pi)), sin((t - 0) * (5 * 2 * pi))]
# end
dt = 0.01
## Rot Stage Arm (in pixel frame)
cr_gt = 1.0
## endeffector pose
start_pose_gt = [0.0, 0.0, 0.0] # x, y, theta

# f = Figure()
# ax = Axis(f[1, 1])
# ts = 0:0.01:4
# lines!(ax, ts, hcat(bootstrap_stepper_u.(ts)...)[1, :])
# lines!(ax, ts, hcat(bootstrap_stepper_u.(ts)...)[2, :])
# lines!(ax, ts, hcat(bootstrap_stepper_u.(ts)...)[3, :])
# display(f)




# Model
window_size = 5000
## Stepper-Inductance System
#### Real System
StepIndSys = Dict(
    :A => [0.0 0 0
        0 0.0 0
        0 0 0.0],
    :B => [1 0 0 # estimated
        0 2 0
        0 0 1],
    :C => [1.0 0 0
        0 1.0 0
        0 0 1.0],
    :D => [0.0 0 0
        0 0 0
        0 0 0],
    :A_m => [0.0 0 0
        0 0.0 0
        0 0 0.0],
    :B_m => [1 0 0
        0 1 0
        0 0 1]
)
StepIndMRAC = Dict(
    :P => lyap((StepIndSys[:A_m] + Diagonal([1.0, 1.0, 1.0]))', -Diagonal([1.0, 1.0, 1.0])),
    :Γ_x => 50.0,
    :Γ_r => 50.0,
    :Γ_w => 10.0,
    :Γ_v => 10.0,
    :Γ_σ => 0.0,
    :Λ => Diagonal([1.0, 1.0, 1.0]),
    :K_x => zeros(3, 3),
    :K_r => Diagonal([1.0, 1.0, 1.0]),
    :W => zeros(10, 3), # (hidden_size, state_size)
    :V => Flux.glorot_uniform(7, 10), # (feature_size, hidden_size)
)
StepIndState = Dict(
    :x => Observable(zeros(3, window_size)),
    :u => Observable(zeros(3, window_size)),
    :r => Observable(zeros(3, window_size)),
    :t => Observable(zeros(window_size)),
    :em => Observable(zeros(3, window_size)), # error of model
)
StepIndKf = KalmanFilter(
    (StepIndSys[:A_m] + Diagonal([1.0, 1.0, 1.0])) * dt,
    StepIndSys[:B_m] * dt,
    StepIndSys[:C],
    StepIndSys[:D],
    cov(MvNormal(3, 0.0001)),
    cov(MvNormal(3, 0.0001)),
    MvNormal(3, 0.0)
)
## Inductance-Pixel System
#### Real System
IndPixSys = Dict(
    :A => [0.0 0 0
        0 0.0 0
        0 0 0.0],
    :B => [1 0 0 # estimated
        0 1 0
        0 0 1],
    :C => [1.0 0 0
        0 1.0 0
        0 0 1.0],
    :D => [0.0 0 0
        0 0 0
        0 0 0],
    :A_m => [0.0 0 0
        0 0.0 0
        0 0 0.0],
    :B_m => [1 0 0
        0 1 0
        0 0 1]
)
IndPixMRAC = (
    :P => lyap((StepIndSys[:A_m] + Diagonal([1.0, 1.0, 1.0]))', -Diagonal([1.0, 1.0, 1.0])),
    :Γ_x => 50.0,
    :Γ_r => 50.0,
    :Γ_w => 10.0,
    :Γ_v => 10.0,
    :Γ_σ => 0.0,
    :Λ => Diagonal([1.0, 1.0, 1.0]),
    :K_x => zeros(3, 3),
    :K_r => Diagonal([1.0, 1.0, 1.0]),
    :W => zeros(10, 3), # (hidden_size, state_size)
    :V => Flux.glorot_uniform(7, 10), # (feature_size, hidden_size)
)
IndPixState = Dict(
    :x => Observable(zeros(3, window_size)),
    :u => Observable(zeros(3, window_size)),
    :r => Observable(zeros(3, window_size)),
    :t => Observable(zeros(window_size)),
    :em => Observable(zeros(3, window_size)), # error of model
)
IndPixKf = KalmanFilter(
    (IndPixSys[:A_m] + Diagonal([1.0, 1.0, 1.0])) * dt,
    IndPixSys[:B_m] * dt,
    IndPixSys[:C],
    IndPixSys[:D],
    cov(MvNormal(3, 0.0001)),
    cov(MvNormal(3, 0.0001)),
    MvNormal(3, 0.0)
)
## Rot Stage Arm
RotStageState = Dict(
    :x => Observable(zeros(3, window_size)), # x, y, theta
    :u => Observable(zeros(3, window_size)),
    :r => Observable(zeros(3, window_size)),
    :t => Observable(zeros(window_size)),
    :em => Observable(zeros(3, window_size)), # error of model
)
CircleState = Dict(
    :x => Observable(zeros(4, window_size)), # x, y, theta, cr
    :t => Observable(zeros(window_size)),
    :em => Observable(zeros(4, window_size)), # error of model
)


function chipFK(state)::Vector
    # x of rotation origin in pixel frame
    # y of rotation origin in pixel frame
    # theta of rotation stage in pixel frame
    # cr of rotation stage in pixel frame
    x, y, theta, cr = state[1], state[2], state[3], state[4]
    return [x + cr * cos(theta),
        y + cr * sin(theta),
        theta]
end

function chipFKbatch(x, y, theta, cr)::AbstractMatrix
    # x of rotation origin in pixel frame
    # y of rotation origin in pixel frame
    # theta of rotation stage in pixel frame
    # cr of rotation stage in pixel frame
    return reduce(hcat,
        [x .+ cr .* cos.(theta),
            y .+ cr .* sin.(theta),
            theta])'
end

function estimateB(us, ys)
    dimensions = size(us[1])[1]
    u_ = reduce(vcat, [kron(u, I(dimensions) |> Matrix)' for u in us])
    B_ = reduce(vcat, [y for y in ys])
    return reshape(u_ \ B_, dimensions, dimensions)'
end

function estimateCircle(x, y)
    cA = [
        -2 * x,
        -2 * y,
        ones(size(x))
    ]
    cA = reduce(hcat, cA)
    cb = -x .^ 2 - y .^ 2
    cres = pinv(cA) * cb
    radius = sqrt(cres[1]^2 + cres[2]^2 - cres[3])
    return cres[1], cres[2], radius
end

function mrac(sys, mracparams, r, r_prev, x, x_prev, dt)
    # calculate u
    feat = [x; r; 1]
    u = mracparams[:K_x]' * x + mracparams[:K_r]' * r + mracparams[:W]' * sigmoid(mracparams[:V]' * feat)

    # ideal reference state
    x_m_dot = sys[:A_m] * x_prev + sys[:B_m] * r_prev
    x_m = x_prev + x_m_dot * dt

    # error prop
    e = x - x_m
    feat = [x_prev; r_prev; 1]

    K_ẋ = -mracparams[:Γ_x] * x * e' * sys[:B] * mracparams[:Λ] * mracparams[:P]
    K_ṙ = -mracparams[:Γ_r] * r * e' * sys[:B] * mracparams[:Λ] * mracparams[:P]
    Ẇ = mracparams[:Γ_w] * ((sigmoid(mracparams[:V]' * feat) - ForwardDiff.jacobian(sigmoid, mracparams[:V]' * feat) * (mracparams[:V]' * feat)) * e' * mracparams[:P] * sys[:B] * mracparams[:Λ] + mracparams[:Γ_σ] * mracparams[:W])
    V̇ = mracparams[:Γ_v] * (feat * e' * mracparams[:P] * sys[:B] * mracparams[:Λ] * mracparams[:W]' * ForwardDiff.jacobian(sigmoid, mracparams[:V]' * feat) + mracparams[:Γ_σ] * mracparams[:V])


    mracparams[:K_x] += K_ẋ * dt
    mracparams[:K_r] += K_ṙ * dt
    mracparams[:W] += Ẇ * dt
    mracparams[:V] += V̇ * dt
    return u, e
end



StepIndState[:x][] .= 0.0
StepIndState[:u][] .= 0.0
StepIndState[:r][] .= 0.0
StepIndState[:t][] .= 0.0
StepIndState[:em][] .= 0.0
ts = -6:dt:20
realStepIndB = [1.0 0.0 0.0;
    0.0 1.0 0.0;
    0.0 0.0 1.0]
realIndPixB = [1.0 0.0 0.0;
    0.0 1.0 0.0;
    0.0 0.0 1.0]
# realStepIndB = [1.0 1.0 0.0;
#     0.0 2.0 1.0;
#     0.0 0.0 1.0]
# realIndPixB = [1.0 0.0 0.3;
#     0.0 1.0 0.3;
#     0.2 0.1 1.0]
realRotStage = [1, 1, 0.0, 1.0] # circle_x, circle_y, circle_theta, circle_radius
# CircleState[:x][][:, 1] = realRotStage
for (i, t) in enumerate(ts)
    ## StepInd

    # shift all
    StepIndState[:x][] = circshift(StepIndState[:x][], (0, 1))
    StepIndState[:u][] = circshift(StepIndState[:u][], (0, 1))
    StepIndState[:r][] = circshift(StepIndState[:r][], (0, 1))
    StepIndState[:em][] = circshift(StepIndState[:em][], (0, 1))
    StepIndState[:t][] = circshift(StepIndState[:t][], (1))

    # update t
    StepIndState[:t][][1] = t

    # update x
    x_dot = (StepIndSys[:A] * StepIndState[:x][][:, 2] + realStepIndB * StepIndState[:u][][:, 2])
    x = StepIndState[:x][][:, 2] + dt * x_dot #+ rand(MvNormal(3, 0.4))
    StepIndState[:x][][:, 1] = x #+ rand(MvNormal(3, 0.003))

    # update B
    B_window = 50
    if t > 3.0 || t < 0.0
        curr_rng = 1:min(i, B_window)
        prev_rng = 2:min(i + 1, B_window + 1)
        dts = (StepIndState[:t][][1:B_window] .- StepIndState[:t][][2:B_window+1])
        dts = [dts'; dts'; dts']
        y = eachcol((
            (StepIndState[:x][][:, 1:B_window] .- StepIndState[:x][][:, 2:B_window+1]) .-
            dts .* (StepIndSys[:A] * StepIndState[:x][][:, 2:B_window+1])) ./ dts)
        u = eachcol(StepIndState[:u][][:, 2:B_window+1])
        StepIndSys[:B] = estimateB(u, y)
        # StepIndState[:em][][:, 1] = StepIndState[:x][][:, 1] .- (StepIndState[:x][][:, 2] + StepIndSys[:A] * StepIndState[:x][][:, 2] + StepIndSys[:B] * StepIndState[:u][][:, 2])
    end


    ## Rot Stage
    # shift all
    RotStageState[:x][] = circshift(RotStageState[:x][], (0, 1))
    RotStageState[:u][] = circshift(RotStageState[:u][], (0, 1))
    RotStageState[:r][] = circshift(RotStageState[:r][], (0, 1))
    RotStageState[:em][] = circshift(RotStageState[:em][], (0, 1))
    RotStageState[:t][] = circshift(RotStageState[:t][], (1))

    # update t
    RotStageState[:t][][1] = t

    # update x
    u = (StepIndState[:x][][:, 2] .- StepIndState[:x][][:, 1]) / dt
    x_dot = (IndPixSys[:A] * IndPixState[:x][][:, 1] + realIndPixB * u)
    IndPixState_x_gt = IndPixState[:x][][:, 1] + dt * x_dot #+ rand(MvNormal(3, 0.4))
    if t > 2.0 && t < 3.0
        IndPixState_x_gt[1:2, 1] .= 0.0
    end
    # IndPixState[:x][][:, 1] = x + rand(MvNormal(3, 0.003))  
    chip_pose = chipFK([
        IndPixState_x_gt[1, 1] + realRotStage[1],
        IndPixState_x_gt[2, 1] + realRotStage[2],
        IndPixState_x_gt[3, 1] + realRotStage[3],
        realRotStage[4]])
    RotStageState[:x][][:, 1] = chip_pose# + rand(MvNormal(3, 0.003))


    ## IndPix
    # shift all
    IndPixState[:x][] = circshift(IndPixState[:x][], (0, 1))
    IndPixState[:u][] = circshift(IndPixState[:u][], (0, 1))
    IndPixState[:r][] = circshift(IndPixState[:r][], (0, 1))
    IndPixState[:em][] = circshift(IndPixState[:em][], (0, 1))
    IndPixState[:t][] = circshift(IndPixState[:t][], (1))

    # update t
    IndPixState[:t][][1] = t

    # update x
    if t > 3.0
        IndPixState[:x][][:, 1] = [
            RotStageState[:x][][1, 1] - CircleState[:x][][1, 1] - CircleState[:x][][4, 1] * cos(RotStageState[:x][][3, 1] + CircleState[:x][][3, 1]),
            RotStageState[:x][][2, 1] - CircleState[:x][][2, 1] - CircleState[:x][][4, 1] * sin(RotStageState[:x][][3, 1] + CircleState[:x][][3, 1]),
            RotStageState[:x][][3, 1] - CircleState[:x][][3, 1]
        ]
    else
        IndPixState[:x][][:, 1] = IndPixState[:x][][:, 2]
    end

    # update B
    B_window = 50
    if t > 3.0
        curr_rng = 1:min(i, B_window)
        prev_rng = 2:min(i + 1, B_window + 1)
        dts = (IndPixState[:t][][1:B_window] .- IndPixState[:t][][2:B_window+1])
        dts = [dts'; dts'; dts']
        y = eachcol((
            (IndPixState[:x][][:, 1:B_window] .- IndPixState[:x][][:, 2:B_window+1]) .-
            dts .* (IndPixSys[:A] * IndPixState[:x][][:, 2:B_window+1])) ./ dts)
        u = eachcol(IndPixState[:u][][:, 2:B_window+1])
        IndPixSys[:B] = estimateB(u, y)

        IndPixState[:em][][:, 1] = IndPixState[:x][][:, 1] .- (IndPixState[:x][][:, 2] + IndPixSys[:A] * IndPixState[:x][][:, 2] + IndPixSys[:B] * IndPixState[:u][][:, 2])
    end

    ## Circle State
    # shift all
    CircleState[:x][] = circshift(CircleState[:x][], (0, 1))
    CircleState[:em][] = circshift(CircleState[:em][], (0, 1))
    CircleState[:t][] = circshift(CircleState[:t][], (1))

    # update t
    CircleState[:t][][1] = t

    # update x
    rotfilter = (RotStageState[:t][] .> 2.0) .& (RotStageState[:t][] .< 3.0)
    rotfilter[1] = false
    if sum(rotfilter) > 20
        # Linear Regression
        circlex, circley, circler = estimateCircle(
            RotStageState[:x][][1, rotfilter] .- IndPixState[:x][][1, rotfilter],
            RotStageState[:x][][2, rotfilter] .- IndPixState[:x][][2, rotfilter])
        CircleState[:x][][:, 1] = [circlex, circley, 0.0, circler]

        # Nonlinear Regression
        model = function (t, p)
            x, y, theta = t[1, :], t[2, :], t[3, :]
            cx, cy, ctheta, cr = p
            return vec(chipFKbatch(
                x .+ cx,
                y .+ cy,
                theta .+ ctheta,
                cr))
        end
        CircleState[:x][][:, 1] = curve_fit(model, IndPixState[:x][][:, rotfilter], vec(RotStageState[:x][][:, rotfilter]), CircleState[:x][][:, 1]).param
    else
        CircleState[:x][][:, 1] = CircleState[:x][][:, 2]
    end
    # CircleState[:x][][:, 1] = realRotStage

    ## Update Controls
    if t < 5.0
        # Calibration mode
        StepIndState[:r][][:, 1] = bootstrap_stepper_u(t)
        StepIndState[:u][][:, 1], StepIndState[:em][][:, 1] = mrac(
            StepIndSys,
            StepIndMRAC,
            StepIndState[:r][][:, 1],
            StepIndState[:r][][:, 2],
            StepIndState[:x][][:, 1],
            StepIndState[:x][][:, 2],
            0.01)
        IndPixState[:u][][:, 1] = StepIndState[:x][][:, 1] .- StepIndState[:x][][:, 2]
    else
        # IndPix control
        target_pose = [0.5, 0.5, 1.57]
        current_pose = RotStageState[:x][][:, 1]

        RotStageState[:em][][:, 1] = (target_pose - current_pose)

        # IndPix control
        try
            IndPixState[:r][][:, 1] = IndPixState[:u][][:, 1] = (
                pinv(ForwardDiff.jacobian(chipFK, [
                IndPixState[:x][][1, 1] + CircleState[:x][][1, 1],
                IndPixState[:x][][2, 1] + CircleState[:x][][2, 1],
                IndPixState[:x][][3, 1] + CircleState[:x][][3, 1],
                CircleState[:x][][4, 1]
            ]))*RotStageState[:em][][:, 1]
            )[1:3]

        catch
            println("jacobian error")
            break
        end


        # StepInd control
        target_ind = StepIndState[:x][][:, 1] .+ pinv(realIndPixB) * IndPixState[:u][][:, 1] * dt
        StepIndState[:r][][:, 1] = -500.0 * (target_ind .- StepIndState[:x][][:, 1])
        StepIndState[:u][][:, 1] = StepIndState[:r][][:, 1]
        StepIndState[:u][][:, 1], StepIndState[:em][][:, 1] = mrac(
            StepIndSys,
            StepIndMRAC,
            StepIndState[:r][][:, 1],
            StepIndState[:r][][:, 2],
            StepIndState[:x][][:, 1],
            StepIndState[:x][][:, 2],
            dt)
    end
end

IndPixSys[:B]

RotStageState[:x][][1, 1]
IndPixState[:x][][1, 1]
StepIndSys[:B]
StepIndState[:r][][:, 1]

f = Figure()

ax = Axis(f[1, 1], title="StepIndState[:r]")
lines!(ax, StepIndState[:t][], StepIndState[:r][][1, :])
lines!(ax, StepIndState[:t][], StepIndState[:r][][2, :])
lines!(ax, StepIndState[:t][], StepIndState[:r][][3, :])

ax = Axis(f[1, 2], title="StepIndState[:u]")
lines!(ax, StepIndState[:t][], StepIndState[:u][][1, :])
lines!(ax, StepIndState[:t][], StepIndState[:u][][2, :])
lines!(ax, StepIndState[:t][], StepIndState[:u][][3, :])

ax = Axis(f[1, 3], title="StepIndState[:x]")
lines!(ax, StepIndState[:t][], StepIndState[:x][][1, :])
lines!(ax, StepIndState[:t][], StepIndState[:x][][2, :])
lines!(ax, StepIndState[:t][], StepIndState[:x][][3, :])

ax = Axis(f[1, 4], title="StepIndState[:em]")
lines!(ax, StepIndState[:t][], StepIndState[:em][][1, :])
lines!(ax, StepIndState[:t][], StepIndState[:em][][2, :])
lines!(ax, StepIndState[:t][], StepIndState[:em][][3, :])

ax = Axis(f[2, 1], title="IndPixState[:r]")
lines!(ax, IndPixState[:t][], IndPixState[:r][][1, :])
lines!(ax, IndPixState[:t][], IndPixState[:r][][2, :])
lines!(ax, IndPixState[:t][], IndPixState[:r][][3, :])

ax = Axis(f[2, 2], title="IndPixState[:u]")
lines!(ax, IndPixState[:t][], IndPixState[:u][][1, :])
lines!(ax, IndPixState[:t][], IndPixState[:u][][2, :])
lines!(ax, IndPixState[:t][], IndPixState[:u][][3, :])

ax = Axis(f[2, 3], title="IndPixState[:x]")
lines!(ax, IndPixState[:t][], IndPixState[:x][][1, :])
lines!(ax, IndPixState[:t][], IndPixState[:x][][2, :])
lines!(ax, IndPixState[:t][], IndPixState[:x][][3, :])

ax = Axis(f[2, 4], title="IndPixState[:em]")
lines!(ax, IndPixState[:t][], IndPixState[:em][][1, :])
lines!(ax, IndPixState[:t][], IndPixState[:em][][2, :])
lines!(ax, IndPixState[:t][], IndPixState[:em][][3, :])

ax = Axis(f[3, 1], title="RotStageState[:x]")
lines!(ax, RotStageState[:t][], RotStageState[:x][][1, :], label="x")
lines!(ax, RotStageState[:t][], RotStageState[:x][][2, :], label="y")
lines!(ax, RotStageState[:t][], RotStageState[:x][][3, :], label="theta")

ax = Axis(f[3, 2], title="RotStageState[:em]")
lines!(ax, RotStageState[:t][], RotStageState[:em][][1, :], label="x")
lines!(ax, RotStageState[:t][], RotStageState[:em][][2, :], label="y")
lines!(ax, RotStageState[:t][], RotStageState[:em][][3, :], label="theta")
axislegend(ax, merge=true, unique=true)

ax = Axis(f[3, 3], title="CircleState[:x]")
lines!(ax, CircleState[:t][], CircleState[:x][][1, :], label="x")
lines!(ax, CircleState[:t][], CircleState[:x][][2, :], label="y")
lines!(ax, CircleState[:t][], CircleState[:x][][3, :], label="r")
axislegend(ax, merge=true, unique=true)

ax = Axis(f[3, 4], aspect=DataAspect())
# color rainbow
scatter!(ax, RotStageState[:x][][1, :], RotStageState[:x][][2, :], color=RotStageState[:t][], colormap=:rainbow)
arrows!(ax, RotStageState[:x][][1, :], RotStageState[:x][][2, :],
    0.01 .* cos.(RotStageState[:x][][3, :]),
    0.01 .* sin.(RotStageState[:x][][3, :]), color=RotStageState[:t][], colormap=:rainbow)
display(f)
