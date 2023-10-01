from stage.sanga import SangaStage, SangaDeltaStage
ss = SangaDeltaStage(port = "/dev/ttyACM1")


# Move stepper motor directly [A, B, C] steps
ss.move_rel_delta([200, 200, 0])

# Move cartesian coordinate relative [X, Y, Z]
ss.move_abs([200, 200, 0])

# Get current position
curr_position = ss.position

# Move cartesian coordinate relative [X, Y, Z]
ss.move_abs([curr_position[0] + 200, curr_position[1] + 200, curr_position[2]])