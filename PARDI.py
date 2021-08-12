# Copyright (c) 2021 Sebastian Pilarski and Slawomir Pilarski
# Patent Pending

import numpy as np
import copy
from optimal import OptimalPolicy

class PARDI:
    """ Predictive Algorithm Reducing Delay Impact (PARDI)
    
        Implements Choose(t)
    """
    def __probs(self, s: int, f: int, u: int, alf0: int, bet0: int, O: np.array):
        """Calculates probability for PARDI choose

        Args:
            s (int):      Number of successes
            f (int):      Number of failures
            u (int):      Number of unknowns
            alf0 (int):   alpha prior
            bet0 (int):   beta prior
            O (np.array): Storage array - output
        """
        s += alf0
        f += bet0
        n  = s + f
        A  = np.empty(u+3)
        
        R = A if u % 2 else O # Read  array
        W = O if u % 2 else A # Write array
        R[0] = 1

        for i in range(u):
            for j in range(i+2): W[j] = 0
            for j in range(i+1): 
                r = R[j]
                W[j+1] += r * (s+j)   / (n+i)
                W[j+0] += r * (f+i-j) / (n+i)
            R, W = W, R

    def choose(self, algo, st: [int], ft: [int], ut: [int], at: [int], bt: [int]) -> int:
        """Chooses the arm via PARDI and provided algorithm

        Args:
            algo ([type]): Algorithm which implements evaluate_arm
            st ([int]):    Successes on each earm
            ft ([int]):    Failures on each arm
            ut ([int]):    Unknowns on each arm
            at ([int]):    alpha prior on each arm
            bt ([int]):    beta prior on each arm

        Returns:
            int: Arm to pull
        """
        # NOTE: T = sum of all successes, failures, unknowns
        assert(len(st) == len(ft) == len(ut) == len(at) == len(bt))
        K = len(st)
        A = np.empty((K, 30)); 
        for i in range(K): self.__probs(st[i], ft[i], ut[i], at[i], bt[i], A[i])

        To = np.array(ut)
        I  = np.zeros(K, dtype=int  )
        P  = np.zeros(K, dtype=float)

        loc = 0
        while True:
            if I[loc] > To[loc]:
                if loc == 0: break
                for i in range(loc, K): I[i] = 0
                loc -= 1
                I[loc] += 1
                continue
            
            k = K - loc
            if k == 1:
                r = 1
                for i in range(K): r *= A[i][I[i]]

                def evc(i: int, eva_s: int, eva_f: int):
                    return   (st[i]+  I[i]      +at[i]) / (st[i]+ft[i]+ut[i]+at[i]+bt[i]) * (1+eva_s) \
                           + (ft[i]+(ut[i]-I[i])+bt[i]) / (st[i]+ft[i]+ut[i]+at[i]+bt[i]) * (0+eva_f)

                ma  = -1
                mev = -1

                for i in range(K):
                    st_s = copy.deepcopy(st); st_f = copy.deepcopy(st)
                    ft_s = copy.deepcopy(ft); ft_f = copy.deepcopy(ft)
                    ut_s = copy.deepcopy(ut); ut_f = copy.deepcopy(ut)
                    at_s = copy.deepcopy(at); at_f = copy.deepcopy(at)
                    bt_s = copy.deepcopy(bt); bt_f = copy.deepcopy(bt)

                    st_s[i] += 1 
                    ft_f[i] += 1

                    eva_s = algo.eval_arm(i, st_s + I, ft_s + (ut_s - I), at_s, bt_s)
                    eva_f = algo.eval_arm(i, st_f + I, ft_f + (ut_s - I), at_f, bt_f)

                    eva = evc(i, eva_s, eva_f)
                    P[i] += r*eva
                
                I[loc] += 1
                continue

            loc += 1

        ma = -1
        mw = -1

        for i in range(K): 
            if P[i] > mw: 
                ma = i
                mw = P[i]
        return ma

                    
def main():
    """Example of running PARDI
    """
    s = [0, 1, 0]
    f = [1, 0, 1]
    u = [1, 3, 1]
    alf0 = 1
    bet0 = 1
    H = 10

    pardi = PARDI()
    opt = OptimalPolicy(3, H, alf0, bet0, 1)
    print("Select arm:", pardi.choose(opt, s, f, u, [1,1,1], [1,1,1]))
                    
main()
                    
                    
