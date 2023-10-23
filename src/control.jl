using LibSerialPort
using GLMakie
using LinearAlgebra
using ProgressMeter
using Serialization
using Distributions
using LowLevelParticleFilters
using LinearAlgebra
using Flux
using ForwardDiff
GLMakie.activate!(inline=false)

portname1 = "/dev/ttyACM0"
portname2 = "/dev/ttyACM1"
baudrate = 115200

function joint_vel_pwm(joint_vel, dt)
    # convert number of steps per second to number of cycles per steps
    # given cps, cycles per second
    # 1 cycle = 1 step
    # the greater the number of cycles per steps, the slower the motor
    # the greater the joint_vel, the smaller the number of cycles per steps, the faster the motor
    joint_vel = clamp(joint_vel, -3, 3)
    cps = 1 / dt
    if joint_vel == 0
        return 10000000000000
    end
    return round(Int, cps / joint_vel)
end

dt = -1
x0 = zeros(6)
LibSerialPort.open(portname1, baudrate) do sp1
    LibSerialPort.open(portname2, baudrate) do sp2
        start_t = time_ns()
        samples = 100
        global x0
        global dt

        for i in 1:samples
            # measure dt
            write(sp1, "mr 10000000000000 10000000000000 10000000000000\n")
            write(sp2, "mr 10000000000000 10000000000000 10000000000000\n")
            while bytesavailable(sp1) < 1
            end
            try
                sp1_data = String(readline(sp1, keep=false))
                ind0, ind1 = split(sp1_data, ',')
                x0[1] = (parse(Int, ind0))
                x0[3] = (parse(Int, ind1))
                print((parse(Int, ind0)) - x00)
            catch
            end
            while bytesavailable(sp2) < 1
            end
            try
                x0[5] = (parse(Int, String(readline(sp2, keep=false))))
            catch
            end
        end
        dt = (time_ns() - start_t) / samples / 1e9
    end
end



window_size = 6000
is = Observable(zeros(window_size, 6))
xs = @lift($is[:, 1])
ẋs = @lift($is[:, 2])
ys = @lift($is[:, 3])
ẏs = @lift($is[:, 4])
zs = @lift($is[:, 5])
żs = @lift($is[:, 6])

# control input, [A, Ȧ, B, Ḃ, C, Ċ]
us = Observable(zeros(window_size, 6))
uA = @lift($us[:, 1])
uȦ = @lift($us[:, 2])
uB = @lift($us[:, 3])
uḂ = @lift($us[:, 4])
uC = @lift($us[:, 5])
uĊ = @lift($us[:, 6])

# adaptive
x_n_log = Observable(zeros(window_size, 6))
u_n_log = Observable(zeros(window_size, 3))
e_log = Observable(zeros(window_size, 6))
x_n_log_x = @lift($x_n_log[:, 1])
x_n_log_y = @lift($x_n_log[:, 3])
x_n_log_z = @lift($x_n_log[:, 5])
u_n_log_A = @lift($u_n_log[:, 1])
u_n_log_B = @lift($u_n_log[:, 2])
u_n_log_C = @lift($u_n_log[:, 3])
e_log_x = @lift($e_log[:, 1])
e_log_y = @lift($e_log[:, 3])
e_log_z = @lift($e_log[:, 5])
nn_loss = Observable(zeros(window_size))

f = Figure()
axx = Axis(f[1, 1], xlabel="x")
lines!(axx, 1:window_size, xs, color=:blue)
lines!(axx, 1:window_size, ys, color=:red)
lines!(axx, 1:window_size, zs, color=:black)
axẋ = Axis(f[1, 2], xlabel="ẋ")
lines!(axẋ, 1:window_size, ẋs, color=:blue)
lines!(axẋ, 1:window_size, ẏs, color=:red)
lines!(axẋ, 1:window_size, żs, color=:black)
axu = Axis(f[2, 1], xlabel="u")
lines!(axu, 1:window_size, uA, color=:blue)
lines!(axu, 1:window_size, uB, color=:red)
lines!(axu, 1:window_size, uC, color=:black)
axu̇ = Axis(f[2, 2], xlabel="u̇")
lines!(axu̇, 1:window_size, uȦ, color=:blue)
lines!(axu̇, 1:window_size, uḂ, color=:red)
lines!(axu̇, 1:window_size, uĊ, color=:black)
axadapt_x = Axis(f[3, 1], xlabel="x_n")
lines!(axadapt_x, 1:window_size, x_n_log_x, color=:blue)
lines!(axadapt_x, 1:window_size, x_n_log_y, color=:red)
lines!(axadapt_x, 1:window_size, x_n_log_z, color=:black)
axadapt_u = Axis(f[3, 2], xlabel="u_n")
lines!(axadapt_u, 1:window_size, u_n_log_A, color=:blue)
lines!(axadapt_u, 1:window_size, u_n_log_B, color=:red)
lines!(axadapt_u, 1:window_size, u_n_log_C, color=:black)
axadapt_e = Axis(f[4, 1], xlabel="e")
lines!(axadapt_e, 1:window_size, e_log_x, color=:blue)
lines!(axadapt_e, 1:window_size, e_log_y, color=:red)
lines!(axadapt_e, 1:window_size, e_log_z, color=:black)
axadapt_nn = Axis(f[4, 2], xlabel="nn_loss")
lines!(axadapt_nn, 1:window_size, nn_loss, color=:blue)
display(f)

# 0,1:B 2,3:C 4,5:A

# Refernce state transition
A = [1 dt 0 0 0 0
    0 1 0 0 0 0
    0 0 1 dt 0 0
    0 0 0 1 0 0
    0 0 0 0 1 dt
    0 0 0 0 0 1]
# Refernce control input
B = [0.5*dt^2 0 0
    dt 0 0
    0 0.5*dt^2 0
    0 dt 0
    0 0 0.5*dt^2
    0 0 dt]
C = [1.0 0 0 0 0 0
    0 0 1.0 0 0 0
    0 0 0 0 1.0 0]
D = [0.0 0 0
    0 0 0
    0 0 0]

dw = MvNormal(6, 1.0)          # Dynamics noise Distribution
de = MvNormal(3, 1.0)          # Measurement noise Distribution
d0 = MvNormal(6, 1.0)   # Initial state Distribution
kf = KalmanFilter(A, B, C, D, cov(dw), cov(de), d0)

MvNormal([1, 1, 1], 1.0)

# define trajectory, accelerating to 0.1 from t = 1 to t = 2, then decelerating to 0 from t = 3 to t = 4
traj_(t, start_t, axis) = begin
    t_, start_t_ = t / 1e9, start_t / 1e9
    # returns acceleraation u and command velocity
    a, b, c, d, e, f, g, h, i = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    target_vel = -5
    pos, vel, acc = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    if t_ - start_t_ > 0.0
    end
    if t_ - start_t_ > a
        # ramp up
        acc[axis] = target_vel / b
        pos[axis] += 0.5 * acc[axis] * (t_ - (start_t_ + a))^2
        vel[axis] += acc[axis] * (t_ - (start_t_ + a))
    end
    if t_ - start_t_ > a + b
        # constant velocity
        acc[axis] = 0.0
        pos[axis] += target_vel * (t_ - (start_t_ + a + b))
        vel[axis] = target_vel
    end
    if t_ - start_t_ > a + b + c
        # ramp down
        acc[axis] = -target_vel / d
        pos[axis] += target_vel * (t_ - (start_t_ + a + b + c)) + 0.5 * acc[axis] * (t_ - (start_t_ + a + b + c))^2
        vel[axis] += acc[axis] * (t_ - (start_t_ + a + b + c))
    end
    if t_ - start_t_ > a + b + c + d
        # stop
        acc[axis] = 0.0
        pos[axis] += 0.0
        vel[axis] = 0.0
    end
    if t_ - start_t_ > a + b + c + d + e
        # ramp up reverse
        acc[axis] = -target_vel / f
        pos[axis] += 0.5 * acc[axis] * (t_ - (start_t_ + a + b + c + d + e))^2
        vel[axis] += acc[axis] * (t_ - (start_t_ + a + b + c + d + e))
    end
    if t_ - start_t_ > a + b + c + d + e + f
        # constant velocity reverse
        acc[axis] = 0.0
        pos[axis] += -target_vel * (t_ - (start_t_ + a + b + c + d + e + f))
        vel[axis] = -target_vel
    end
    if t_ - start_t_ > a + b + c + d + e + f + g
        # ramp down reverse
        acc[axis] = target_vel / h
        pos[axis] += -target_vel * (t_ - (start_t_ + a + b + c + d + e + f + g)) + 0.5 * acc[axis] * (t_ - (start_t_ + a + b + c + d + e + f + g))^2
        vel[axis] += acc[axis] * (t_ - (start_t_ + a + b + c + d + e + f + g))
    end
    if t_ - start_t_ > a + b + c + d + e + f + g + h
        # stop
        acc[axis] = 0.0
        pos[axis] += 0.0
        vel[axis] = 0.0
    end
    if t_ - start_t_ > a + b + c + d + e + f + g + h + i
        return nothing, nothing, nothing
    end
    return pos, vel, acc
end

traj(t, start_t, axis) = begin
    # track sine wave velocity, 0.5 Hz
    t_, start_t_ = t / 1e9, start_t / 1e9
    # returns acceleraation u and command velocity
    acc = [0.0, 0.0, 0.0]
    pos = [0.0, 0.0, 0.0]
    vel = [0.0, 0.0, 0.0]

    mag = 4
    freq = 0.2
    vel[axis] = mag * sin(2 * pi * freq * (t_ - start_t_))
    acc[axis] = mag * 2 * pi * freq * cos(2 * pi * freq * (t_ - start_t_))
    pos[axis] = mag / (2 * pi * freq) * cos(2 * pi * freq * (t_ - start_t_))
    if t_ - start_t_ > 5
        vel[axis] = 0.0
        acc[axis] = 0.0
    end
    if t_ - start_t_ > 6.5
        return nothing, nothing, nothing
    end
    return pos, vel, acc
end


# LibSerialPort.open(portname1, baudrate) do sp1
#     LibSerialPort.open(portname2, baudrate) do sp2
#         global W, V, K_x, K_r, x_n, u_n, x_m, Λ, A_n, B_n, dt, Γ_x, Γ_r, Γ_w, Γ_V, Γ_σ, P, A_m, B_m, x0, is, us, x_n_log, u_n_log, e_log, Θ, ϕ, opt_state, x_m_track, u_m
#         while bytesavailable(sp1) > 0
#             read(sp1)
#         end
#         while bytesavailable(sp2) > 0
#             read(sp2)
#         end

#         vel = [0.0, 0.0, 0.0]
#         acc = [0.0, 0.0, 0.0]
#         init = true
#         target_state = nothing
#         while true
#             if init
#                 write(sp1, "mr 10000000000000 10000000000000 10000000000000\n")
#                 write(sp2, "mr 10000000000000 10000000000000 10000000000000\n")
#             end
#             is[] = circshift(is[], (1, 0))
#             is[][1, 1] = is[][2, 1]
#             is[][1, 2] = is[][2, 2]

#             while bytesavailable(sp1) < 1
#             end
#             try
#                 sp1_data = String(readline(sp1, keep=false))
#                 ind0, ind1 = split(sp1_data, ',')
#                 is[][1, 1] = ((parse(Int, ind0)) - x0[1]) * 1.0e-7
#                 is[][1, 3] = ((parse(Int, ind1)) - x0[3]) * 1.0e-7
#             catch
#             end
#             while bytesavailable(sp2) < 1
#             end
#             try
#                 is[][1, 5] = ((parse(Int, String(readline(sp2, keep=false)))) - x0[5]) * 1.0e-7
#             catch
#             end

#             kf([0, 0, 0], [is[][1, 1], is[][1, 3], is[][1, 5]])
#             is[][1, 1] = state(kf)[1]
#             is[][1, 2] = state(kf)[2]
#             is[][1, 3] = state(kf)[3]
#             is[][1, 4] = state(kf)[4]
#             is[][1, 5] = state(kf)[5]
#             is[][1, 6] = state(kf)[6]
#             notify(is)
#             us[] = circshift(us[], (1, 0))
#             us[][1, 1] = vel[1]
#             us[][1, 2] = acc[1]
#             us[][1, 3] = vel[2]
#             us[][1, 4] = acc[2]
#             us[][1, 5] = vel[3]
#             us[][1, 6] = acc[3]
#             notify(us)
#             yield()

#             if init
#                 init = false
#                 target_state = is[][1, :]
#             end

#             vel .= 1000 * (is[][1, :] .- target_state)[[1, 3, 5]]
#             e_log[] = circshift(e_log[], (1, 0))
#             e_log[][1, :] = (is[][1, :] .- target_state)
#             notify(e_log)

#             sp1_cmd = "x\n"
#             sp2_cmd = "mr $(joint_vel_pwm(vel[1], dt)) $(joint_vel_pwm(vel[2], dt)) $(joint_vel_pwm(vel[3], dt))\n"
#             write(sp1, sp1_cmd)
#             write(sp2, sp2_cmd)
#         end

#     end
# end



LibSerialPort.open(portname1, baudrate) do sp1
    LibSerialPort.open(portname2, baudrate) do sp2
        global W, V, K_x, K_r, x_n, u_n, x_m, Λ, A_n, B_n, dt, Γ_x, Γ_r, Γ_w, Γ_V, Γ_σ, P, A_m, B_m, x0, is, us, x_n_log, u_n_log, e_log, Θ, ϕ, opt_state, x_m_track, u_m
        while bytesavailable(sp1) > 0
            read(sp1)
        end
        while bytesavailable(sp2) > 0
            read(sp2)
        end

        # Implements Model-Refernce Adaptive Control (MRAC) using 
        # a linear reference model in the joint space
        # and single-layer neural network for Disturbance and Uncertainty Model
        # see http://liberzon.csl.illinois.edu/teaching/ece517notes-post.pdf
        # see https://www.mathworks.com/help/slcontrol/ug/model-reference-adaptive-control.html
        # p127** https://www.cds.caltech.edu/archive/help/uploads/wiki/files/140/IEEE_WorkShop_Slides_Lavretsky.pdf

        # Refernce model in joint space theta (ẋ_m = Am xm + Bm um)
        # this is from the inductance sensor output
        x_m = zeros(3) # current state, [position A, velocity A, position B, velocity B, position C, velocity C]
        # Refernce state transition
        A_m = [1.0 0 0
            0 1.0 0
            0 0 1.0]
        # Refernce control input
        B_m = [dt 0 0
            0 dt 0
            0 0 dt]

        # define lyapunov function as V = 1/2 * x^T * P * x
        # state cost for lyapunov function
        Q = Diagonal([0.4, 0.4, 0.4]) # position cost is 100 times velocity cost
        ## Solve the P matrix, which is the solution to the Lyapunov equation:
        ## A^T * P + P * A = -Q
        P = lyap(A_m', -Q)

        # learning rate for weight update
        Γ_x = 1
        Γ_r = 1
        Γ_w = 0.1
        Γ_V = 0.1
        Γ_θ = 1
        Γ_σ = 0.0 # sigma modification to add damping

        # ϕ(x) = σ(V^T * x)
        feature_size = 16
        ϕ = Chain(Dense(10, feature_size, σ), Dense(feature_size, feature_size, σ))
        opt_state = Flux.setup(Adam(0.1), ϕ)

        # initial parameters

        # Adaptive parameters
        ## Nominal stepper linear model ẋ = A_n x_n + B_n Λ (u_n - f(x))
        ## KNOWN measured state
        x_n = zeros(3) # current state, [position A, velocity A, position B, velocity B, position C, velocity C]
        u_n = zeros(3) # current control input, [pwm A, pwm B, pwm C]
        Λ = Diagonal([1, 1, 1]) # check p80
        ## UNKNOWN state transition
        A_n = [1.0 0 0
            0 1.0 0
            0 0 1.0]
        ## UNKNOWN control input
        B_n = [dt 0 0
            0 dt 0
            0 0 dt]
        ## UNKNOWN gains of nominal u_n
        ## u_n = K_x' * x_n + K_r' * r_n + W' * phi(x_n)
        K_x = zeros(3, 3)
        K_r = zeros(3, 3)
        W = zeros(feature_size, 3)
        V = zeros(10, feature_size)
        Θ = zeros(feature_size, 3)

        # target joint state
        x_m = copy(x_n)
        x_m_track = copy(x_n)
        # x_m_track[1] += 0.002 * 1.0e7
        x_m_track[1:end] .= 0.0
        u_m = zeros(3)

        for i = 1:100
            t_prev = Int64(time_ns())
            t_start = Int64(time_ns())
            pos, vel, acc = traj(time_ns(), t_start, i % 3 + 1)
            loopcnt = 0
            while true
                loopcnt += 1
                if isnothing(vel)
                    break
                end
                sp1_cmd = "x\n"
                sp2_cmd = "mr $(joint_vel_pwm(vel[1], dt)) $(joint_vel_pwm(vel[2], dt)) $(joint_vel_pwm(vel[3], dt))\n"
                write(sp1, sp1_cmd)
                write(sp2, sp2_cmd)

                is[] = circshift(is[], (1, 0))
                is[][1, 1] = is[][2, 1]
                is[][1, 2] = is[][2, 2]

                while bytesavailable(sp1) < 1
                end
                try
                    sp1_data = String(readline(sp1, keep=false))
                    ind0, ind1 = split(sp1_data, ',')
                    is[][1, 1] = ((parse(Int, ind0)) - x0[1]) * 1.0e-7
                    is[][1, 3] = ((parse(Int, ind1)) - x0[3]) * 1.0e-7
                catch
                end
                while bytesavailable(sp2) < 1
                end
                try
                    is[][1, 5] = ((parse(Int, String(readline(sp2, keep=false)))) - x0[5]) * 1.0e-7
                catch
                end

                kf([0, 0, 0], [is[][1, 1], is[][1, 3], is[][1, 5]])
                is[][1, 1] = state(kf)[1]
                is[][1, 2] = state(kf)[2]
                is[][1, 3] = state(kf)[3]
                is[][1, 4] = state(kf)[4]
                is[][1, 5] = state(kf)[5]
                is[][1, 6] = state(kf)[6]
                notify(is)
                us[] = circshift(us[], (1, 0))
                us[][1, 1] = vel[1]
                us[][1, 2] = acc[1]
                us[][1, 3] = vel[2]
                us[][1, 4] = acc[2]
                us[][1, 5] = vel[3]
                us[][1, 6] = acc[3]
                notify(us)
                yield()

                # Loop
                if false
                    # training mode
                    u_n = [vel[1], vel[2], vel[3]]
                    feat = [is[][1, :]; u_n; 1]
                    # training
                    grads = Flux.gradient(ϕ) do m
                        result = m(feat)
                        sum(((A_n * x_n + B_n * Λ * (u_n - Θ' * result)) .- x_m) .^ 2)
                    end
                    Flux.update!(opt_state, ϕ, grads[1])

                    # update
                    x_n_old = copy(x_n)
                    x_n = A_n * x_n + B_n * Λ * (u_n - Θ' * ϕ(feat))
                    x_m = is[][1, 1:2:end]

                    e = x_n - x_m
                    K̇_x = -Γ_x * (x_n * e' * P * B_n * Λ + Γ_σ * K_x)
                    K̇_r = -Γ_r * (u_n * e' * P * B_n * Λ + Γ_σ * K_r)
                    Θ̇ = -Γ_θ * (ϕ(feat) * e' * P * B_n * Λ + Γ_σ * Θ)

                    K_x += K̇_x * dt
                    K_r += K̇_r * dt
                    Θ += Θ̇ * dt

                    # nominal control input
                    u_m = 1000 * (is[][1, 1:2:end].-x_m_track)
                    u_n_ = K_x' * x_n + K_r' * u_m + Θ' * ϕ(feat)
                    x_n_ = A_n * x_n + B_n * Λ * (u_n - Θ' * ϕ(feat))

                    x_n = copy(x_m)

                    x_n_log[] = circshift(x_n_log[], (1, 0))
                    u_n_log[] = circshift(u_n_log[], (1, 0))
                    e_log[] = circshift(e_log[], (1, 0))
                    nn_loss[] = circshift(nn_loss[], (1))
                    x_n_log[][1, 1:2:end] = x_n_
                    u_n_log[][1, :] = u_n_
                    e_log[][1, 1:2:end] = e
                    nn_loss[][1] = sum(((A_n * x_n + B_n * Λ * (u_n - Θ' * ϕ(feat))) .- x_m) .^ 2)
                    notify(x_n_log)
                    notify(u_n_log)
                    notify(e_log)

                    if i < 10
                        pos, vel, acc = traj(time_ns(), t_start, i % 3 + 1)
                        x_m_track[1:end] = is[][1, 1:2:end]
                        x_m_track[1] += 0.023
                    else
                        Γ_x = 1
                        Γ_r = 1
                        Γ_w = 0.1
                        Γ_V = 0.1
                        vel .= u_n_
                    end
                else
                    # switch to tracking mode
                    # update
                    u_n = [vel[1], vel[2], vel[3]]
                    feat = [is[][1, :]; u_n; 1]
                    x_n_old = copy(x_n)
                    x_n = A_n * x_n + B_n * Λ * (u_n - W' * sigmoid(V' * feat))
                    x_m = is[][1, 1:2:end]
                    e = x_n - x_m

                    K̇_x = Γ_x * (x_n * e' * P * B_n + Γ_σ * K_x)
                    K̇_r = Γ_r * (u_n * e' * P * B_n + Γ_σ * K_r)
                    Ẇ = Γ_w * ((sigmoid(V' * feat) - ForwardDiff.jacobian(sigmoid, V' * feat) * (V' * feat)) * e' * P * B_n + Γ_σ * W)
                    V̇ = Γ_V * (feat * e' * P * B_n * W' * ForwardDiff.jacobian(sigmoid, V' * feat) + Γ_σ * V)

                    K_x += K̇_x
                    K_r += K̇_r
                    W += Ẇ
                    V += V̇

                    x_n = copy(x_m)


                    # nominal control input
                    u_m = 1000 * (is[][1, 1:2:end].-x_m_track)
                    u_n_ = K_x' * x_n + K_r' * u_m + W' * sigmoid(V' * feat)
                    # nominal state
                    u_ad = W' * sigmoid(V' * feat)
                    x_n_ = A_n * x_n + B_n * Λ * (u_n - u_ad)

                    x_n_log[] = circshift(x_n_log[], (1, 0))
                    u_n_log[] = circshift(u_n_log[], (1, 0))
                    e_log[] = circshift(e_log[], (1, 0))
                    nn_loss[] = circshift(nn_loss[], (1))
                    x_n_log[][1, 1:2:end] = x_n
                    u_n_log[][1, :] = W' * sigmoid(V' * feat)
                    e_log[][1, 1:2:end] =  (is[][1, 1:2:end].-x_m_track)
                    nn_loss[][1] =
                        notify(x_n_log)
                    notify(u_n_log)
                    notify(e_log)

                    if i < 4
                        pos, vel, acc = traj(time_ns(), t_start, i % 3 + 1)
                        x_m_track[1:end] = is[][1, 1:2:end]
                        x_m_track[1] += 0.023
                    else
                        Γ_x = 1
                        Γ_r = 1
                        Γ_w = 0.01
                        Γ_V = 0.01
                        vel .= u_m
                    end
                end
            end
            dt = (Int64(time_ns()) - t_prev) / 1e9 / loopcnt
            write(sp2, "mr 10000000000000 10000000000000 10000000000000\n")
        end
    end
end




# training
grads = Flux.gradient(ϕ) do m
    result = m(feat)
    sum(((A_n * x_n + B_n * Λ * (u_n - Θ' * result)) .- x_m) .^ 2)
end
Flux.update!(opt_state, ϕ, grads[1])

# update
x_n = A_n * x_n + B_n * Λ * (u_n - Θ' * ϕ(feat))
x_m = is[][1, :]
u_n = [acc[1], acc[2], acc[3]]

e = x_n - x_m
K̇_x = -Γ_x * (x_n * e' * P * B_n * Λ + Γ_σ * K_x)
K̇_r = -Γ_r * (u_n * e' * P * B_n * Λ + Γ_σ * K_r)
Θ̇ = -Γ_θ * (ϕ(feat) * e' * P * B_n * Λ + Γ_σ * Θ)

K_x += K̇_x
K_r += K̇_r
Θ += Θ̇





# Implements Model-Refernce Adaptive Control (MRAC) using 
# a linear reference model in the joint space
# and single-layer neural network for Disturbance and Uncertainty Model
# see http://liberzon.csl.illinois.edu/teaching/ece517notes-post.pdf
# see https://www.mathworks.com/help/slcontrol/ug/model-reference-adaptive-control.html
# p127** https://www.cds.caltech.edu/archive/help/uploads/wiki/files/140/IEEE_WorkShop_Slides_Lavretsky.pdf

# Refernce model in joint space theta (ẋ_m = Am xm + Bm um)
# this is from the inductance sensor output
x_m = zeros(6) # current state, [position A, velocity A, position B, velocity B, position C, velocity C]
# Refernce state transition
A_m = [1 dt 0 0 0 0
    0 1 0 0 0 0
    0 0 1 dt 0 0
    0 0 0 1 0 0
    0 0 0 0 1 dt
    0 0 0 0 0 1]
# Refernce control input
B_m = [0.5*dt^2 0 0
    dt 0 0
    0 0.5*dt^2 0
    0 dt 0
    0 0 0.5*dt^2
    0 0 dt]

# define lyapunov function as V = 1/2 * x^T * P * x
# state cost for lyapunov function
Q = Diagonal([1, 0.01, 1, 0.01, 1, 0.01]) # position cost is 100 times velocity cost
## Solve the P matrix, which is the solution to the Lyapunov equation:
## A^T * P + P * A = -Q
P = lyap(A_m', -Q)

# learning rate for weight update
Γ_x = 0.1
Γ_r = 0.1
Γ_w = 0.1
Γ_V = 0.1
Γ_σ = 0 # sigma modification to add damping

# learnable weights for the single layer neural network
# phi(x) = σ(V^T * x)
layer_width = 10
feature_size = 7 # use [x_n] as input + 1 for bias
W = zeros(layer_width, 3)
V = zeros(feature_size, layer_width)
sigmoid(x) = 1 ./ (1 .+ exp.(-x))

# initial parameters

# Adaptive parameters
## Nominal stepper linear model ẋ = A_n x_n + B_n Λ (u_n - f(x))
## KNOWN measured state
x_n = zeros(6) # current state, [position A, velocity A, position B, velocity B, position C, velocity C]
u_n = zeros(3) # current control input, [pwm A, pwm B, pwm C]
Λ = Diagonal([1, 1, 1]) # check p80
## UNKNOWN state transition
A_n = [1 dt 0 0 0 0
    0 1 0 0 0 0
    0 0 1 dt 0 0
    0 0 0 1 0 0
    0 0 0 0 1 dt
    0 0 0 0 0 1]
## UNKNOWN control input
B_n = [0.5*dt^2 0 0
    dt 0 0
    0 0.5*dt^2 0
    0 dt 0
    0 0 0.5*dt^2
    0 0 dt]
## UNKNOWN gains of nominal u_n
## u_n = K_x' * x_n + K_r' * r_n + W' * phi(x_n)
K_x = zeros(6, 3)
K_r = zeros(3, 3)


# Loop

# nominal state
x_n = is[][1, :]

# target joint state
x_m = copy(x_n)
# x_m[1] += 0.002 * 1.0e7
# x_m[2:2:end] .= 0.0
u_m = zeros(3)

# update
e = x_n - x_m

K̇_x = -Γ_x * (x_n * e' * P * B_n + Γ_σ * K_x)
K̇_r = -Γ_r * (u_n * e' * P * B_n + Γ_σ * K_r)
Ẇ = Γ_w * ((sigmoid(V' * [x_n; 1]) - ForwardDiff.jacobian(sigmoid, V' * [x_n; 1]) * (V' * [x_n; 1])) * e' * P * B_n + Γ_σ * W)
V̇ = Γ_V * ([x_n; 1] * e' * P * B_n * W' * ForwardDiff.jacobian(sigmoid, V' * [x_n; 1]) + Γ_σ * V)

K_x += K̇_x * dt
K_r += K̇_r * dt
W += Ẇ * dt
V += V̇ * dt

# nominal control input
u_n = K_x' * x_n + K_r' * u_m + W' * sigmoid(V' * [x_n; 1])
# u_n = clamp.(u_n, -0.1, 0.1)
# nominal state
u_ad = W' * sigmoid(V' * [x_n; 1])
x_n_ = A_n * x_n + B_n * Λ * (u_n - u_ad)

x_n_ - x_m


