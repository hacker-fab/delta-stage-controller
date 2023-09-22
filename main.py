# %% start
from stage.sanga import SangaStage, SangaDeltaStage
ss = SangaDeltaStage(port = "/dev/ttyACM1")

# %% move
ss.move_rel(200, axis = "z")

# %%
ss.board.query("i?")
# %%
