# Copyright (c) 2021 Sebastian Pilarski and Slawomir Pilarski

import optPol.OptPol as OptPol

class OptimalPolicy:
    """ Wrapper for optimal policy for Bernoulli bandits 

        Implements eval_arm
    """
    def __init__(self, k, H, alf, bet, nthrds):
        print("Calculating optimal policy...")
        OptPol.compute_expected(k, H, alf, bet, nthrds)
    
    def eval_arm(self, i, st, ft, at, bt):
        if len(st) == 2: return OptPol.OptPol_value2(int(st[0]), int(ft[0]), int(st[1]), int(ft[1]))
        if len(st) == 3: return OptPol.OptPol_value3(int(st[0]), int(ft[0]), int(st[1]), int(ft[1]), int(st[2]), int(ft[2]))
        else:
            print("Wrapper for any value of k-arms not implemented - can be added in cpp source and rebuilt")
            return -1
            

