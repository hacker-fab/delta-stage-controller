from stage.sanga import SangaStage, SangaDeltaStage
ss = SangaDeltaStage(port = "/dev/ttyACM1")


ss.move_rel_delta([200, 200, 0])
curr_position = ss.position
ss.move_abs([curr_position[0] + 200, curr_position[1] + 200, curr_position[2]])