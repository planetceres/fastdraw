
MSG = ""
def print_wipe(msg1, msg0=False):
    backspace(len(MSG))
    global MSG
    if (msg0 == True):
        MSG=""
    else:
        MSG = msg1
    print(msg1, end="")
    return

def backspace(n):
    print('\r' * n, end="")
    print(" " * len(MSG), end="")
    print('\r' * n, end="")