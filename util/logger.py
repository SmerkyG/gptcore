from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only

class Logger():
    log_level = 0

def log_always(*args):
    rank_zero_info(args)

def log(*args):
    if Logger.log_level > 0:
        rank_zero_info(args)
