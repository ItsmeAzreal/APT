import math

def check_loss(loss, step=None):
    if not math.isfinite(loss.item()):
        msg = f"Loss NaN/inf detected{' at step ' + str(step) if step is not None else ''}!"
        print(msg)
        return False
    return True
