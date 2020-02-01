import math
import numpy as np
from itertools import chain, combinations


def subsets(iterable, fromsize, tosize):
    s = list(iterable)
    step = 1 + (fromsize > tosize) * -2
    return chain.from_iterable(combinations(s, r)
                               for r in range(fromsize, tosize + step, step))


def nCr(n, r):
    if r > n:
        return 0
    f = math.factorial
    return f(n) // f(r) // f(n-r)


# https://stackoverflow.com/questions/46374185/does-python-have-a-function-which-computes-multinomial-coefficients
def multinomial(lst):
    res, i = 1, 1
    for a in lst:
        for j in range(1, a+1):
            res *= i
            res //= j
            i += 1
    return res


def partitions(n):
    div_sets = list(subsets(range(1, n), 0, n-1))
    for divs in div_sets:
        partition = list()
        start = 0
        if divs:
            for end in divs:
                partition.append(list(range(n))[start:end])
                start = end
            partition.append(list(range(n))[end:])
        else:
            partition.append(list(range(n)))
        yield partition


def n_dags(n_nodes, max_indegree):
    n_cache = list()
    for m in range(n_nodes + 1):
        if m < 2:
            n_cache.append(1)
            continue
        res = 0
        for k in range(1, m+1):
            term = 0
            for d in range(max_indegree + 1):
                term += nCr(m-k, d)
            term = term ** k
            res += (-1) ** (k-1) * nCr(m, k) * term * n_cache[m - k]
        n_cache.append(res)
    return n_cache[n_nodes]


def read_jkl(scorepath):
    scores = dict()
    with open(scorepath, 'r') as jkl_file:
        rows = jkl_file.readlines()
        scores = dict()
        n_scores = 0
        for row in rows[1:]:

            if not n_scores:
                n_scores = int(row.strip().split()[1])
                current_var = int(row.strip().split()[0])
                scores[current_var] = dict()
                continue

            row_list = row.strip().split()
            score = float(row_list[0])
            n_parents = int(row_list[1])
            
            parents = frozenset()
            if n_parents > 0:
                parents = frozenset([int(x) for x in row_list[2:]])
            scores[current_var][frozenset(parents)] = score
            n_scores -= 1

    return scores


def gen_M_layerings(n, M):
    for B in partitions(n):
        if len(B) == 1:
            yield B
            continue
        psums = list()
        for i in range(1, len(B)):
            psums.append(len(B[i-1]) + len(B[i]))
        if min(psums) <= M:
            continue
        else:
            yield B


def map_names(names, B):
    new_B = list()
    for layer in B:
        new_layer = set()
        for v in layer:
            new_layer.add(names[v])
        new_B.append(frozenset(new_layer))
    return new_B


def logminus(p1, p2):
    # p1 bigger log prob
    # p2 smaller log prob

    if p1-p2 == 0:
        # Is this ok?
        # I suppose it doesn't hurt even if we falsely go to the final return?
        # This doesn't return false positives?
        return -float('inf')
    return max(p1, p2) + np.log1p(-np.exp(-abs(p1-p2)))


def parentsums(B, M, max_indegree, pi_v):
    # redundancy in having both the indegree and pi_v precomputed
    tau = dict({v: dict() for v in set().union(*B)})
    tau_hat = dict({v: dict() for v in set().union(*B)})

    B.append(set())
    for j in range(len(B)-1):
        B_to_j = set().union(*B[:j+1])
        for v in B[j+1]:
            tau_hat[v][frozenset({v})] = -float('inf')
            for G_v in subsets(B_to_j, 1, max_indegree):
                if frozenset(G_v).intersection(B[j]):
                    tau_hat[v][frozenset({v})] = np.logaddexp(tau_hat[v][frozenset({v})], pi_v[v][frozenset(G_v)])
        if len(B[j]) <= M:
            for v in B[j].union(B[j+1]):
                for C in subsets(B[j], 0, len(B[j])):
                    tau[v][frozenset(C)] = -float('inf')
                for G_v in subsets(B_to_j.difference({v}), 0, max_indegree):
                    tau[v][frozenset(G_v).intersection(B[j])] = np.logaddexp(tau[v][frozenset(G_v).intersection(B[j])], pi_v[v][frozenset(G_v)])
                for D in subsets(B[j].difference({v}), 0, len(B[j].difference({v}))):
                    tau_hat[v][frozenset(D)] = -float('inf')
                    for C in subsets(D, 0, len(D)):
                        tau_hat[v][frozenset(D)] = np.logaddexp(tau_hat[v][frozenset(D)], tau[v][frozenset(C)])
    B.pop()  # drop the added empty set

    return tau_hat


def pi_B(B, M, max_indegree, pi_v):

    tau_hat = parentsums(B, M, max_indegree, pi_v)

    l = len(B)  # l is last proper index
    B = [frozenset()] + B + [frozenset()]
    g = {i: dict() for i in range(0, l+2)}
    p = dict()
    f = dict()

    for j in range(l, -1, -1):

        if len(B[j]) > M or j == 0:
            P = [(B[j], B[j])]
        else:                                                  # 5
            P = list()
            for D in subsets(B[j], len(B[j]), 1):
                for T in subsets(D, 1, len(D)):
                    P.append((frozenset(D), frozenset(T)))

        for DT in P:                                           # 6
            D, T = DT
            if D == B[j]:
                j_prime = j + 1
                D_prime = frozenset()
            else:
                j_prime = j
                D_prime = D

            if D == B[j] and j < l and len(B[j+1]) > M:        # 10
                S = [B[j+1]]
                A = B[j+1].difference({min(B[j+1])})
            else:                                              # 12
                S = [frozenset(Si) for Si in subsets(B[j_prime].difference(D_prime), 1, max(1, len(B[j_prime].difference(D_prime))))]
                A = frozenset()

            # p = dict()
            for v in B[j_prime]:                               # 13
                if not T:
                    p[v] = pi_v[v][frozenset()]
                else:
                    if T == B[j]:
                        p[v] = tau_hat[v][frozenset({v})]       # 17
                    else:
                        p[v] = logminus(tau_hat[v][D.difference({v})], tau_hat[v][D.difference({v}).difference(T)])

            # f = dict()
            f[A] = 0                                            # 19
            for v in A:
                f[A] += p[v]

            if D not in g[j]:
                g[j][D] = dict()
            if not S:                                          # 22
                g[j][D][T] = 0
            else:
                g[j][D][T] = -float('inf')

            for Si in S:                                       # 27
                v = min(Si)
                f[Si] = f[Si.difference({v})] + p[v]
                if D != B[j] or j == 0 or len(Si) > M - len(B[j]):
                    g[j][D][T] = np.logaddexp(g[j][D][T], f[Si] + g[j_prime][D_prime.union(Si)][Si])

    return g[0][frozenset()][frozenset()]


def valid_sources(b, M):
    # undefined if b, M is invalid
    valids = np.array([False]*len(b))
    for j in range(len(b)):
        if j == 0 or j == len(b)-1:
            valids[j] = True
            continue
        if b[j] > 1:
            if (b[j]-1 + b[j+1] > M and b[j]-1 + b[j-1] > M-1) or (b[j]-1 + b[j+1] > M-1 and b[j]-1 + b[j-1] > M):
                # in first case have to add v to prev, in second case have to add v to next
                valids[j] = True
        else:
            # if len(B[j]) == 1 layer disappears
            valids[j] = True

    return valids


def valid_targets(b, j, M):

    # undefined if j is invalid source

    valids = np.array([False]*(2*len(b)+1))
    b_stripped = [b[i] if i != j else b[i]-1 for i in range(len(b))]

    # 1. forced to place v in certain layer
    for i in range(len(b_stripped)-1):
        if b_stripped[i] + b_stripped[i+1] <= M and b_stripped[i] != 0 and b_stripped[i+1] != 0:
            if i == j:
                valids[(i+2)*2-1] = True
            else:
                valids[(i+1)*2-1] = True
            return valids

    # 2. possibly multiple options where to place v
    sizes = [0]
    for s in b_stripped:
        sizes.append(s)
        sizes.append(0)
    for i in range(len(sizes)):
        if i+1 == (j+1)*2:
            continue
        if i == 0:  # first possible new layer
            if sizes[i] + sizes[i+1] >= M:  # uusi layeri
                valids[i] = True
        elif i == len(sizes) - 1:  # last possible new layer
            if sizes[i] + sizes[i-1] >= M:
                valids[i] = True
        elif i % 2 == 0:  # middle new layers
            if sizes[i] + sizes[i-1] >= M and sizes[i] + sizes[i+1] >= M:
                valids[i] = True
        elif i % 2 != 0:  # existing layers
            if i == 1:  # first existing layer
                if sizes[i] + sizes[i+2] >= M:
                    valids[i] = True
            elif i == len(sizes)-2:  # last existing layer
                if sizes[i] + sizes[i-2] >= M:
                    valids[i] = True
            else:  # middle layers
                if sizes[i] + sizes[i-2] >= M and sizes[i] + sizes[i+2]:
                    valids[i] = True

    return valids


def MCMC_chain(B, M, t_max, max_indegree, scores):
    states = [B]
    probs = [pi_B(B, M, max_indegree, scores)]

    for t in range(t_max):

        stay_prob = 0.01

        if np.random.rand() < stay_prob:
            states.append(B)
            probs.append(probs[-1])

        else:

            b = [len(layer) for layer in B]

            possible_sources = valid_sources(b, M)
            source = np.random.choice(np.nonzero(possible_sources)[0])

            possible_targets = valid_targets(b, source, M)
            target = np.random.choice(np.nonzero(possible_targets)[0])

            v = np.random.choice(list(B[source]))

            B_prime = [layer.difference({v}) for layer in B]

            # Add v to target
            rev_source = 0
            if target % 2 == 1:  # add to existing layer
                B_prime[target//2] = B_prime[target//2].union({v})
                rev_source = target//2
            else:  # add to new layer
                if target == 0:
                    B_prime = [frozenset({v})] + B_prime
                    rev_source = 0
                else:  # new layer after target//2
                    B_prime = B_prime[0:target//2+1] + [frozenset({v})] + B_prime[target//2+1:]
                    rev_source = target//2

            # delete possible 0 layers and adjust rev_source accordingly
            for i in range(len(B_prime)):
                if len(B_prime[i]) == 0:
                    del B_prime[i]
                    if i < rev_source:
                        rev_source -= 1
                    break  # there can be max 1 0-layer

            b_prime = [len(layer) for layer in B_prime]

            q = 1/(sum(possible_sources)*b[source]*sum(possible_targets))
            q_rev = 1/(sum(valid_sources(b_prime, M))*b_prime[rev_source]*sum(valid_targets(b_prime, rev_source, M)))

            prob_B = probs[-1]
            prob_B_prime = pi_B(B_prime, M, max_indegree, scores)

            if np.random.rand() < min(1, np.exp(prob_B_prime - prob_B)*q_rev/q):
                states.append(B_prime)
                probs.append(prob_B_prime)
                B = B_prime
            else:
                states.append(B)
                probs.append(prob_B)

    return probs, states
