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
GLMakie.activate!(inline=false)


# Simulation
# bootstrap_stepper_u = function (t)
#     if t >= 0.0 && t < 1.0
#         return [sin((t - 0) * (2 * pi)), 0.0, 0.0]
#     elseif t >= 1.0 && t < 2.0
#         return [0.0, sin((t - 1) * (2 * pi)), 0.0]
#     elseif t >= 2.0 && t < 3.0
#         return [0.0, 0.0, sin((t - 2) * (2 * pi))]
#     else
#         return [0.0, 0.0, 0.0]
#     end
# end
bootstrap_stepper_u = function (t)
    return [sin((t - 0) * (5 * 2 * pi)), cos((t - 0) * (5 * 2 * pi)), sin((t - 0) * (5 * 2 * pi))]
end

## Stepper-Inductance System B (x, y, theta)
B_s_gt = [1.0 0.0 0.0;
    0.0 1.0 0.0;
    0.0 0.0 1.0]
## Inductance-Pixel System B (x, y, theta)
B_p_gt = [1.0 0.0 0.0;
    0.0 1.0 0.0;
    0.0 0.0 1.0]
## Rot Stage Arm (in pixel frame)
cr_gt = 1.0
## endeffector pose
start_pose_gt = [0.0, 0.0, 0.0] # x, y, theta

f = Figure()
ax = Axis(f[1, 1])
ts = 0:0.01:4
lines!(ax, ts, hcat(bootstrap_stepper_u.(ts)...)[1, :])
lines!(ax, ts, hcat(bootstrap_stepper_u.(ts)...)[2, :])
lines!(ax, ts, hcat(bootstrap_stepper_u.(ts)...)[3, :])
display(f)



# Model
window_size = 2000
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
    :Γ_w => 5.0,
    :Γ_v => 5.0,
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
    :Γ_w => 5.0,
    :Γ_v => 5.0,
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
## Rot Stage Arm
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

function estimateB(us, ys)
    dimensions = size(us[1])[1]
    u_ = reduce(vcat, [kron(u, I(dimensions) |> Matrix)' for u in us])
    B_ = reduce(vcat, [y for y in ys])
    return reshape(u_ \ B_, dimensions, dimensions)'
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
dt = 0.01
ts = 0:dt:10
realB = [10.0 2.0 1.0;
    5.0 1.0 3.0;
    -1.0 4.0 1.0]
for (i, t) in enumerate(ts)                                      

    # shift all
    StepIndState[:x][] = circshift(StepIndState[:x][], (0, 1))
    StepIndState[:u][] = circshift(StepIndState[:u][], (0, 1))
    StepIndState[:r][] = circshift(StepIndState[:r][], (0, 1))
    StepIndState[:em][] = circshift(StepIndState[:em][], (0, 1))
    StepIndState[:t][] = circshift(StepIndState[:t][], (1))

    # update t
    StepIndState[:t][][1] = t

    # update x
    x_dot = (StepIndSys[:A] * StepIndState[:x][][:, 2] + realB * StepIndState[:u][][:, 2])
    StepIndState[:x][][:, 1] = StepIndState[:x][][:, 2] + dt * x_dot 

    # update r
    StepIndState[:r][][:, 1] = bootstrap_stepper_u(t)
    StepIndState[:u][][:, 1] = StepIndState[:r][][:, 1]
    StepIndState[:u][][:, 1], StepIndState[:em][][:, 1] = mrac(
        StepIndSys,
        StepIndMRAC,
        StepIndState[:r][][:, 1],
        StepIndState[:r][][:, 2],
        StepIndState[:x][][:, 1],
        StepIndState[:x][][:, 2],
        0.01)

    if i > 3
        y = eachcol(((StepIndState[:x][][:, 1:i-1] .- StepIndState[:x][][:, 2:i]) .- dt * StepIndSys[:A] * StepIndState[:x][][:, 2:i]) ./ dt)
        u = eachcol(StepIndState[:u][][:, 2:i])
        StepIndSys[:B] = estimateB(u, y)
    end
end

f = Figure()

axu = Axis(f[1, 1])
lines!(axu, StepIndState[:t][], StepIndState[:r][][1, :])
lines!(axu, StepIndState[:t][], StepIndState[:r][][2, :])
lines!(axu, StepIndState[:t][], StepIndState[:r][][3, :])

axu = Axis(f[2, 1])
lines!(axu, StepIndState[:t][], StepIndState[:u][][1, :])
lines!(axu, StepIndState[:t][], StepIndState[:u][][2, :])
lines!(axu, StepIndState[:t][], StepIndState[:u][][3, :])

axx = Axis(f[3, 1])
lines!(axx, StepIndState[:t][], StepIndState[:x][][1, :])
lines!(axx, StepIndState[:t][], StepIndState[:x][][2, :])
lines!(axx, StepIndState[:t][], StepIndState[:x][][3, :])

axem = Axis(f[4, 1])
lines!(axem, StepIndState[:t][], StepIndState[:em][][1, :])
lines!(axem, StepIndState[:t][], StepIndState[:em][][2, :])
lines!(axem, StepIndState[:t][], StepIndState[:em][][3, :])
display(f)

StepIndSys[:B]





StepIndState[:x][][:, 2:5] + dt * (StepIndSys[:A] * StepIndState[:x][][:, 2:5] +
                                   estimateB(u, y) * StepIndState[:u][][:, 2:5])
StepIndState[:x][][:, 1:4]





StepIndState[:x][][3, :]
StepIndState[:x][][1, :]
StepIndState[:r][][3, :]
StepIndState[:t][]


merge(fff, (a=2,))

-np.dot(Gamma_x, np.dot(x, e.T).dot(B_hat).dot(Lambda).dot(P))

sigmoid

# Stage Control
# inputs: x, y, theta
# parameters: B matrix
stage_B = [1.0 0.0 0.0;
    0.0 1.0 0.0;
    0.0 0.0 1.0]

# outputs: Forward Kinematics
# parameters: cx, cy, cr (circle center, radius)
stage_center = [1.0, 1.0, 1.0]
function staggeFK(state)::Vector
    x, y, theta = state[1], state[2], state[3]
    stage_center = state[4:end]
    return [x + stage_center[1] + stage_center[3] * cos(theta),
        y + stage_center[2] + stage_center[3] * sin(theta),
        theta]
end

# target_pose
target_pose = [0.0, 1.0, 0.0]
current_pose = [0.0, 0.0, 0.0]
for i in 1:10
    u = pinv(ForwardDiff.jacobian(staggeFK, [current_pose; stage_center])) * (target_pose - current_pose)
    current_pose = u[1:3] + current_pose
    println(current_pose)
end
