import math
import numpy as np
from itertools import chain, combinations
import time
from collections import defaultdict


def structure_to_str(structure):
    # Input is a layering, root partition, or DAG
    return '|'.join([' '.join([str(node) for node in part]) for part in structure])


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
            yield map_names(np.random.permutation(n), B)


def map_names(names, B):
    new_B = list()
    for part in B:
        new_layer = set()
        for v in part:
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


def sample_DAG(R, scores, max_indegree):
    DAG = list((root,) for root in R[0])
    tmp_scores = [scores[root][frozenset()] for root in R[0]]
    for j in range(1, len(R)):
        ban = set().union(*R[min(len(R)-1, j):])
        req = R[max(0, j-1)]
        for i in R[j]:
            psets = list()
            pset_scores = list()
            for pset in scores[i].keys():
                if not pset.intersection(ban) and pset.intersection(req) and len(pset) <= max_indegree:
                    psets.append(pset)
                    pset_scores.append(scores[i][pset])
            normalizer = sum(np.exp(pset_scores - np.logaddexp.reduce(pset_scores)))
            k = np.where(np.random.multinomial(1, np.exp(pset_scores - np.logaddexp.reduce(pset_scores))/normalizer))[0][0]
            DAG.append((i,) + tuple(psets[k]))
            tmp_scores.append(pset_scores[k])
            
    return DAG, sum(tmp_scores)


def parentsums(B, M, max_indegree, pi_v):

    tau = dict({v: defaultdict(lambda: -float("inf")) for v in set().union(*B)})
    tau_hat = dict({v: dict() for v in set().union(*B)})

    B.append(set())
    for j in range(len(B)-1):
        B_to_j = set().union(*B[:j+1])
        
        for v in B[j+1]:
            
            tmp_dict = defaultdict(list)
            for G_v in subsets(B_to_j, 1, max_indegree):
                if frozenset(G_v).intersection(B[j]):
                    tmp_dict[v].append(pi_v[v][frozenset(G_v)])
            tau_hat[v].update((frozenset({item[0]}), np.logaddexp.reduce(item[1])) for item in tmp_dict.items())
        
        if len(B[j]) <= M:
            for v in B[j].union(B[j+1]):
                
                tmp_dict = defaultdict(list)
                for G_v in subsets(B_to_j.difference({v}), 0, max_indegree):
                    tmp_dict[frozenset(G_v).intersection(B[j])].append(pi_v[v][frozenset(G_v)])
                
                tau[v].update((item[0], np.logaddexp.reduce(item[1])) for item in tmp_dict.items())
                
                for D in subsets(B[j].difference({v}), 0, len(B[j].difference({v}))):
                    tau_hat[v][frozenset(D)] = np.logaddexp.reduce(np.array([tau[v][frozenset(C)] for C in subsets(D, 0, len(D))]))
                    
    B.pop()  # drop the added empty set

    return tau_hat


def pi_B(B, M, max_indegree, pi_v, tau_hat, return_all=False):

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

            # p = dict() # poistetaan D_prime 3.2.2020
            for v in B[j_prime].difference(D_prime):                      # 13
                if not T:
                    p[v] = pi_v[v][frozenset()]
                else:
                    if T == B[j]:
                        p[v] = tau_hat[v][frozenset({v})]       # 17
                    else:
                        p[v] = logminus(tau_hat[v][D], tau_hat[v][D.difference(T)])

            # f = dict()
            f[A] = 0                                            # 19
            for v in A:
                f[A] += p[v]

            if D not in g[j]:
                g[j][D] = dict()
            if not S:                                          # 22
                g[j][D][T] = 0.0
            else:
                g[j][D][T] = -float('inf')

            tmp_list = list([g[j][D][T]])
            for Si in S:                                       # 27
                v = min(Si)
                f[Si] = f[Si.difference({v})] + p[v]
                if D != B[j] or j == 0 or len(Si) > M - len(B[j]):
                    tmp_list.append(f[Si] + g[j_prime][D_prime.union(Si)][Si])
            
            g[j][D][T] = np.logaddexp.reduce(tmp_list)        

    if return_all:
        return g
    return g[0][frozenset()][frozenset()]


def valid_layering(B, M):
    if min([len(B[j]) for j in range(len(B))]) == 0:
        return False
    if len(B) == 1:
        return True
    psums = list()
    for j in range(1, len(B)):
        psums.append(len(B[j-1]) + len(B[j]))
    if min(psums) <= M:
        return False
    return True


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
            # if len(b[j]) == 1 part disappears
            valids[j] = True

    return valids


def valid_targets(b, j, M):

    # undefined if j is invalid source

    valids = np.array([False]*(2*len(b)+1))
    b_stripped = [b[i] if i != j else b[i]-1 for i in range(len(b))]

    # 1. forced to place v in certain part
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
        if i == 0:  # first possible new part
            if sizes[i] + sizes[i+1] >= M:  # uusi layeri
                valids[i] = True
        elif i == len(sizes) - 1:  # last possible new part
            if sizes[i] + sizes[i-1] >= M:
                valids[i] = True
        elif i % 2 == 0:  # middle new layers
            if sizes[i] + sizes[i-1] >= M and sizes[i] + sizes[i+1] >= M:
                valids[i] = True
        elif i % 2 != 0:  # existing layers
            if i == 1:  # first existing part
                if sizes[i] + sizes[i+2] >= M:
                    valids[i] = True
            elif i == len(sizes)-2:  # last existing part
                if sizes[i] + sizes[i-2] >= M:
                    valids[i] = True
            else:  # middle layers
                if sizes[i] + sizes[i-2] >= M and sizes[i] + sizes[i+2]:
                    valids[i] = True

    return valids


def possible_psets(U, T, max_indegree):
    if not U:
        yield frozenset()
    for required in subsets(T, 1, max(1, max_indegree)):
        for additional in subsets(U.difference(T), 0, max_indegree - len(required)):
            yield frozenset(required).union(additional)


def score_v(v, pset, scores):
    return scores[v][pset]


def hat_pi(v, U, T, scores, max_indegree):
    return np.logaddexp.reduce([score_v(v, pset, scores) for pset in possible_psets(U, T, max_indegree)])


def f(U, T, S, scores, max_indegree):
    hat_pi_sum = 0
    for v in S:
        hat_pi_sum += hat_pi(v, U, T, scores, max_indegree)
    return hat_pi_sum


def pi_R(R, scores, max_indegree):
    f_sum = 0
    for i in range(len(R)):
        f_sum += f(set().union(*R[:i]),
                    [R[i-1] if i-1>-1 else set()][0],
                    R[i], scores, max_indegree)
    return f_sum


def R_to_B(R, M):
    B = list()
    B_layer = R[0]
    for i in range(1, len(R)):
        if len(B_layer) + len(R[i]) <= M:
            B_layer = B_layer.union(R[i])
        else:
            B.append(B_layer)
            B_layer = R[i]
    B.append(B_layer)
    return B


def R_basic_move(**kwargs):

    def valid():
        return True

    R = kwargs["R"]

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid()

    m = len(R)
    sum_binoms = [sum([nCr(len(R[i]), c) for c in range(1, len(R[i]))]) for i in range(m)]
    nbd = m - 1 + sum(sum_binoms)
    q = 1/nbd

    j = np.random.choice(range(1, nbd+1))

    R_prime = list()
    if j < m:
        R_prime = [R[i] for i in range(j-1)] + [R[j-1].union(R[j])] + [R[i] for i in range(min(m, j+1), m)]
        return R_prime, q, q

    sum_binoms = [sum(sum_binoms[:i]) for i in range(1, len(sum_binoms)+1)]
    i_star = [m-1 + sum_binoms[i] for i in range(len(sum_binoms)) if m-1 + sum_binoms[i] < j]
    i_star = len(i_star)

    c_star = [nCr(len(R[i_star]), c) for c in range(1, len(R[i_star])+1)]
    c_star = [sum(c_star[:i]) for i in range(1, len(c_star)+1)]

    c_star = [m-1 + sum_binoms[i_star-1] + c_star[i] for i in range(len(c_star))
               if m-1 + sum_binoms[i_star-1] + c_star[i] < j]
    c_star = len(c_star)+1

    nodes = np.random.choice(list(R[i_star]), c_star)

    R_prime = [R[i] for i in range(i_star)] + [frozenset(nodes)]
    R_prime += [R[i_star].difference(nodes)] + [R[i] for i in range(min(m, i_star+1), m)]

    return R_prime, q, q


def R_swap_any(**kwargs):

    def valid():
        return len(R) > 1

    R = kwargs["R"]

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid()

    j, k = np.random.choice(range(len(R)), 2, replace=False)
    v_j = np.random.choice(list(R[j]))
    v_k = np.random.choice(list(R[k]))
    R_prime = list()
    for i in range(len(R)):
        if i == j:
            R_prime.append(R[i].difference({v_j}).union({v_k}))
        elif i == k:
            R_prime.append(R[i].difference({v_k}).union({v_j}))
        else:
            R_prime.append(R[i])

    q = 1/(nCr(len(R), 2)*len(R[j])*len(R[k]))

    return R_prime, q, q


def B_swap_nonadjacent(**kwargs):
    # TODO update docstring to reflect new behaviour
    """Swaps the layer of two nodes in different layers by sampling uniformly at random:
    1. j and k, j != k
    2. nodes in B[j] and B[k]
    and finally swapping the layers of the chosen nodes.

    The proposed layering is valid and cannot be reached by the relocate function with one step.
    The proposal probability is symmetric.

    Args:
       B (list): Initial state of the layering for the relocation transition

    Returns:
        Proposed new M-layering, proposal probability and reverse proposal probability
    """

    def valid():
        return len(B) > 2

    B = kwargs["B"]

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid()

    if len(B) == 3:
        j = np.random.choice([0,2])
    else:
        j = np.random.choice(range(len(B)))
    k = np.random.choice(np.setdiff1d(np.array(range(len(B))), [max(0, j-1), j, min(len(B)-1, j+1)]))
    v_j= np.random.choice(list(B[j]))
    v_k = np.random.choice(list(B[k]))
    B_prime = list()
    for i in range(len(B)):
        if i == j:
            B_prime.append(B[i].difference({v_j}).union({v_k}))
        elif i == k:
            B_prime.append(B[i].difference({v_k}).union({v_j}))
        else:
            B_prime.append(B[i])

    n_opts = 2*(len(B)-2) + (len(B)-2)*(len(B)-3)/2

    q = 1/(n_opts*len(B[j])*len(B[k]))
    return B_prime, q, q


def B_swap_adjacent(**kwargs):
    """Swaps the layer of two nodes in adjacent layers by sampling uniformly at random:
    1. j between 1 and l-1
    2. nodes in B[j] and B[j+1]
    and finally swapping the layers of the chosen nodes.

    The proposed layering is valid and cannot be reached by the relocate function with one step.
    The proposal probability is symmetric.

    Args:
       B (list): Initial state of the layering for the relocation transition

    Returns:
        Proposed new M-layering, proposal probability and reverse proposal probability
    """
    def valid():
        return len(B) > 1

    B = kwargs["B"]

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid()

    j = np.random.randint(len(B) - 1)
    v_1 = np.random.choice(list(B[j]))
    v_2 = np.random.choice(list(B[j+1]))
    B_prime = list()
    for i in range(len(B)):
        if i == j:
            B_prime.append(B[i].difference({v_1}).union({v_2}))
        elif i == j+1:
            B_prime.append(B[i].difference({v_2}).union({v_1}))
        else:
            B_prime.append(B[i])
    q = 1/((len(B)-1)*len(B[j])*len(B[j+1]))
    return B_prime, q, q


def B_relocate_many(**kwargs):

    def valid():
        return len(B) > 1 and len(valid_sources) > 0

    B = kwargs["B"]
    M = kwargs["M"]

    valid_sources = [i for i in range(len(B)) if len(B[i]) > 1]

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid()

    B_prime = [frozenset()]

    source = np.random.choice(valid_sources)
    target = np.random.choice(list(set(range(2*len(B)+1)).difference({source*2+1})))
    size = np.random.randint(2, len(B[source])+1)
    nodes = np.random.choice(list(B[source]), size)

    B_prime = [layer.difference({*nodes}) for layer in B]

    # Add nodes to target
    rev_source = 0
    if target % 2 == 1:  # add to existing part
        B_prime[target//2] = B_prime[target//2].union({*nodes})
        rev_source = target//2
    else:  # add to new part
        if target == 0:
            B_prime = [frozenset({*nodes})] + B_prime
            rev_source = 0
        else:  # new part after target//2
            B_prime = B_prime[0:target//2+1] + [frozenset({*nodes})] + B_prime[target//2+1:]
            rev_source = target//2

    # delete possible 0 layers and adjust rev_source accordingly
    for i in range(len(B_prime)):
        if len(B_prime[i]) == 0:
            del B_prime[i]
            if i < rev_source:
                rev_source -= 1
            break  # there can be max 1 0-part

    valid_rev_sources = [i for i in range(len(B_prime)) if len(B_prime[i]) > 1]
    q = 1/(len(valid_sources)*len(B[source])*(2*len(B)+1))
    q_rev = 1/(len(valid_rev_sources)*len(B_prime[rev_source])*(2*len(B_prime)+1))

    return B_prime, q, q_rev


def B_relocate_one(**kwargs):
    """Relocates a single node in the input M-layering B by choosing uniformly at random:
    1. a source part from among the valid possibilities,
    2. a node within the source part,
    3. a valid target part, including any possible new part, where to move the node.

    In step (1), only such layers are valid, from which it is possible to draw a node,
    and create a valid new M-layering by moving it to another part. In step (3) the possible
    target layers depend on the source part sampled in step (1). The proposed M-layering
    is thus guaranteed to be valid and different from the input M-layering.

    Args:
       B (list): Initial state of the layering for the relocation transition
       M (int):  Specifies the space of Bs

    Returns:
        Proposed new M-layering, proposal probability and reverse proposal probability
    """
    def valid():
        return len(B) > 1

    B = kwargs["B"]
    M = kwargs["M"]

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid()

    b = [len(part) for part in B]
    possible_sources = valid_sources(b, M)
    source = np.random.choice(np.nonzero(possible_sources)[0])
    possible_targets = valid_targets(b, source, M)
    target = np.random.choice(np.nonzero(possible_targets)[0])
    v = np.random.choice(list(B[source]))
    B_prime = [layer.difference({v}) for layer in B]

    # Add v to target
    rev_source = 0
    if target % 2 == 1:  # add to existing part
        B_prime[target//2] = B_prime[target//2].union({v})
        rev_source = target//2
    else:  # add to new part
        if target == 0:
            B_prime = [frozenset({v})] + B_prime
            rev_source = 0
        else:  # new part after target//2
            B_prime = B_prime[0:target//2+1] + [frozenset({v})] + B_prime[target//2+1:]
            rev_source = target//2

    # delete possible 0 layers and adjust rev_source accordingly
    for i in range(len(B_prime)):
        if len(B_prime[i]) == 0:
            del B_prime[i]
            if i < rev_source:
                rev_source -= 1
            break  # there can be max 1 0-part

    b_prime = [len(part) for part in B_prime]

    q = 1/(sum(possible_sources)*b[source]*sum(possible_targets))
    q_rev = 1/(sum(valid_sources(b_prime, M))*b_prime[rev_source]*sum(valid_targets(b_prime, rev_source, M)))

    return B_prime, q, q_rev


def relocate_weighted(B, M):
    """Same as :py:func`relocate_uniform`, but in step (1) the possible source 
    layers are given probability proportional to their size (i.e., number of nodes in them),
    and in step (3) the target layers are given probability proportional to their size+1,
    where +1 is needed to make new layers possible.

    Args:
       B (list): Initial state of the layering for the relocation transition
       M (int):  Specifies the space of Bs

    Returns:
        Proposed new M-layering, proposal probability and reverse proposal probability
    """
    
    b = [len(part) for part in B]        
    possible_sources = valid_sources(b, M)
    possible_sources = np.nonzero(possible_sources)[0]
    n_valid_sources = sum([b[i] for i in possible_sources])
    source = np.random.choice(possible_sources, p=[b[i]/n_valid_sources for i in possible_sources])
    possible_targets = valid_targets(b, source, M)
    possible_targets = np.nonzero(possible_targets)[0]
    target_normalizer = sum([b[i//2]+1 if i % 2 == 1 else 1 for i in possible_targets])
    target = np.random.choice(possible_targets, p=[(b[i//2]+1)/target_normalizer
                                                   if i % 2 == 1 else 1/target_normalizer
                                                   for i in possible_targets])

    v = np.random.choice(list(B[source]))

    B_prime = [part.difference({v}) for part in B]

    # Add v to target
    rev_source = 0
    if target % 2 == 1:  # add to existing part
        B_prime[target//2] = B_prime[target//2].union({v})
        rev_source = target//2
    else:  # add to new part
        if target == 0:
            B_prime = [frozenset({v})] + B_prime
            rev_source = 0
        else:  # new part after target//2
            B_prime = B_prime[0:target//2+1] + [frozenset({v})] + B_prime[target//2+1:]
            rev_source = target//2

    # delete possible 0 layers and adjust rev_source accordingly
    for i in range(len(B_prime)):
        if len(B_prime[i]) == 0:
            del B_prime[i]
            if i < rev_source:
                rev_source -= 1
            break  # there can be max 1 0-part

    b_prime = [len(part) for part in B_prime]

    rev_possible_sources = valid_sources(b_prime, M)
    rev_possible_sources = np.nonzero(rev_possible_sources)[0]
    rev_n_valid_sources = sum([b_prime[i] for i in rev_possible_sources])
    rev_possible_targets = valid_targets(b_prime, rev_source, M)
    rev_possible_targets = np.nonzero(rev_possible_targets)[0]
    rev_target_normalizer = sum([b_prime[i//2] if i % 2 == 1 else 1 for i in rev_possible_targets])

    q = (1/n_valid_sources)*([b[target//2]+1 if target % 2 == 1 else 1][0]/target_normalizer)
    q_rev = (1/rev_n_valid_sources)*(b[source]/rev_target_normalizer)

    return B_prime, q, q_rev


def relocate_weighted_inverse(B, M):
    """Same as :py:func`relocate_uniform`, but in step (1) the possible source 
    layers are given probability proportional to their size (i.e., number of nodes in them),
    and in step (3) the target layers are given probability inversely proportional to their size+1,
    where +1 is needed to avoid 0-division for new layers.

    Args:
       B (list): Initial state of the layering for the relocation transition
       M (int):  Specifies the space of Bs

    Returns:
        Proposed new M-layering, proposal probability and reverse proposal probability
    """

    b = [len(part) for part in B]
    possible_sources = valid_sources(b, M)
    possible_sources = np.nonzero(possible_sources)[0]
    n_valid_sources = sum([b[i] for i in possible_sources])
    source = np.random.choice(possible_sources, p=[b[i]/n_valid_sources for i in possible_sources])
    possible_targets = valid_targets(b, source, M)
    possible_targets = np.nonzero(possible_targets)[0]
    target_normalizer = sum([1/(b[i//2]+1) if i % 2 == 1 else 1 for i in possible_targets])
    target = np.random.choice(possible_targets, p=[(1/(b[i//2]+1))/target_normalizer
                                                   if i % 2 == 1 else 1/target_normalizer
                                                   for i in possible_targets])

    v = np.random.choice(list(B[source]))

    B_prime = [part.difference({v}) for part in B]

    # Add v to target
    rev_source = 0
    if target % 2 == 1:  # add to existing part
        B_prime[target//2] = B_prime[target//2].union({v})
        rev_source = target//2
    else:  # add to new part
        if target == 0:
            B_prime = [frozenset({v})] + B_prime
            rev_source = 0
        else:  # new part after target//2
            B_prime = B_prime[0:target//2+1] + [frozenset({v})] + B_prime[target//2+1:]
            rev_source = target//2

    # delete possible 0 layers and adjust rev_source accordingly
    for i in range(len(B_prime)):
        if len(B_prime[i]) == 0:
            del B_prime[i]
            if i < rev_source:
                rev_source -= 1
            break  # there can be max 1 0-part

    b_prime = [len(part) for part in B_prime]

    rev_possible_sources = valid_sources(b_prime, M)
    rev_possible_sources = np.nonzero(rev_possible_sources)[0]
    rev_n_valid_sources = sum([b_prime[i] for i in rev_possible_sources])
    rev_possible_targets = valid_targets(b_prime, rev_source, M)
    rev_possible_targets = np.nonzero(rev_possible_targets)[0]
    rev_target_normalizer = sum([1/(b_prime[i//2]+1) if i % 2 == 1 else 1 for i in rev_possible_targets])

    q = (1/n_valid_sources)*([1/(b[target//2]+1) if target % 2 == 1 else 1][0]/target_normalizer)
    q_rev = (1/rev_n_valid_sources)*((1/b[source])/rev_target_normalizer)

    return B_prime, q, q_rev


def relocate_validate(B, M):
    """Relocates a single node in the input M-layering B by choosing uniformly at random:
    1. a source part from among every existing part,
    2. a node within the source part,
    3. a target part, including any possible new part, where to move the node.

    Then

    4. If the previous 3 steps produced an invalid M-layering, the layering is made valid
    by combining adjacent layers starting from the beginning.

    In contrast to the other relocate moves, this might then return the same M-layering
    that was given as input.

    Args:
       B (list): Initial state of the layering for the relocation transition
       M (int):  Specifies the space of Bs

    Returns:
        Proposed new M-layering, proposal probability and reverse proposal probability
    """

    b = [len(part) for part in B]
    source = np.random.choice(range(len(B)))
    target = np.random.choice(range(len(B)*2+1))
    v = np.random.choice(list(B[source]))
    B_prime = [part.difference({v}) for part in B]

    # Add v to target
    rev_source = 0
    if target % 2 == 1:  # add to existing part
        B_prime[target//2] = B_prime[target//2].union({v})
        rev_source = target//2
    else:  # add to new part
        if target == 0:
            B_prime = [frozenset({v})] + B_prime
            rev_source = 0
        else:  # new part after target//2
            B_prime = B_prime[0:target//2+1] + [frozenset({v})] + B_prime[target//2+1:]
            rev_source = target//2

    # delete possible 0 layers and adjust rev_source accordingly
    for i in range(len(B_prime)):
        if len(B_prime[i]) == 0:
            del B_prime[i]
            if i < rev_source:
                rev_source -= 1
            break  # there can be max 1 0-part

    # merge adjacent layers if their combined size is less or equal to M
    # and adjust rev_source accordingly
    i = 0
    while i < len(B_prime) - 1:
        j = i+1
        layersum = len(B_prime[i])
        while j < len(B_prime) and layersum + len(B_prime[j]) <= M:
            layersum += len(B_prime[j])
            j += 1
        if layersum != len(B_prime[i]):
            B_prime[i] = frozenset().union(*B_prime[i:j])
            for k in range(j-1-i):
                if i+1 < len(B_prime):
                    del B_prime[i+1]
                if i+1 < rev_source:
                    rev_source -= 1
        i += 1

    # Something wrong with rev_source adjustment above
    # so lets brute-force
    for i in range(len(B_prime)):
        if v in B_prime[i]:
            rev_source = i
        
    b_prime = [len(part) for part in B_prime]

    q = 1/(len(B)*b[source]*(len(B)*2+1))
    # print('B_prime {}'.format(B_prime))
    # print('v {}, rev_source {}'.format(v, rev_source))
    q_rev = 1/(len(B_prime)*b_prime[rev_source]*(len(B_prime)*2+1))

    return B_prime, q, q_rev


def MCMC(M, iterations, max_indegree, scores, print_steps=False, seed=None):

    if seed is not None:
        np.random.seed(seed)

    def print_step(del_first=True):
        format_func = {"B": structure_to_str,
                       "DAG": structure_to_str}
        print(",".join(["{}"]*len(stats_keys)).format(*[stats[key][-1]
                                                        if key not in format_func
                                                        else format_func[key](stats[key][-1])
                                                        for key in stats_keys]))
        if del_first:
            for key in stats_keys:
                del stats[key][0]

    def update_stats(B_prob, DAG_prob, B, DAG,
                     acceptance_prob, accepted, move,
                     t_parentsums, t_posterior):
        stats["B_prob"].append(B_prob)
        stats["B"].append(B)
        stats["DAG"].append(DAG)
        stats["DAG_prob"].append(DAG_prob)
        stats["t_parentsums"].append(t_parentsums)
        stats["t_posterior"].append(t_posterior)
        stats["accepted"].append(accepted)
        stats["acceptance_prob"].append(acceptance_prob)
        stats["move"].append(move)

    if len(scores.keys()) == M:
        B = [frozenset(scores.keys())]
    else:
        B = map_names(list(scores.keys()),
                      np.random.choice(list(gen_M_layerings(len(scores.keys()), M)), 1)[0])

    stay_prob = 0.01

    stats_keys = ["B_prob", "DAG_prob", "B", "DAG",
                  "acceptance_prob", "accepted", "move",
                  "t_parentsums", "t_posterior"]
    stats = {key: list() for key in stats_keys}

    t_psum = time.process_time()
    tau_hat = parentsums(B, M, max_indegree, scores)
    t_psum = time.process_time() - t_psum

    t_pos = time.process_time()
    g = pi_B(B, M, max_indegree, scores, tau_hat, return_all=True)
    B_prob = g[0][frozenset()][frozenset()]
    t_pos = time.process_time() - t_pos

    B_moves = [B_relocate_one, B_relocate_many, B_swap_adjacent, B_swap_nonadjacent]
    R_moves = [R_basic_move, R_swap_any]
    moves = B_moves + R_moves
    moveprob_counts = np.array([10, 10, 10, 10, 5, 5])

    R = generate_partition(B, M, tau_hat, g, scores)
    DAG, DAG_prob = sample_DAG(R, scores, max_indegree)

    update_stats(B_prob, DAG_prob, B, DAG, None, None, None, t_psum, t_pos)

    if print_steps:
        print_step(del_first=False)
        
    for i in range(iterations-1):

        if np.random.rand() < stay_prob:
            R = generate_partition(B, M, tau_hat, g, scores)
            DAG, DAG_prob = sample_DAG(R, scores, max_indegree)
            update_stats(B_prob, DAG_prob, B, DAG, None, None, "stay", None, None)

        else:

            move = np.random.choice(moves, p=moveprob_counts/sum(moveprob_counts))

            if not move(B=B, M=M, R=R, validate=True):
                R = generate_partition(B, M, tau_hat, g, scores)
                DAG, DAG_prob = sample_DAG(R, scores, max_indegree)
                update_stats(B_prob, DAG_prob, B, DAG, None, None, "invalid_input_" + move.__name__, None, None)
                if print_steps:
                    print_step()
                continue

            if move in B_moves:

                B_prime, q, q_rev = move(B=B, M=M)

                B_prob = stats["B_prob"][-1]

                if not valid_layering(B_prime, M):
                    R = generate_partition(B, M, tau_hat, g, scores)
                    DAG, DAG_prob = sample_DAG(R, scores, max_indegree)
                    update_stats(B_prob, DAG_prob, B, DAG, None, None, "invalid_output_" + move.__name__, None, None)
                    if print_steps:
                        print_step()
                    continue

                t_psum = time.process_time()
                tau_hat_prime = parentsums(B_prime, M, max_indegree, scores)
                t_psum = time.process_time() - t_psum

                t_pos = time.process_time()
                g_prime = pi_B(B_prime, M, max_indegree, scores, tau_hat_prime, return_all=True)
                B_prime_prob = g_prime[0][frozenset()][frozenset()]
                t_pos = time.process_time() - t_pos

                acc_prob = np.exp(B_prime_prob - B_prob)*q_rev/q

            elif move in R_moves:

                R_prime, q, q_rev = move(R=R)
                B_prime = R_to_B(R_prime, M)

                if B_prime == B:
                    R = generate_partition(B, M, tau_hat, g, scores)
                    DAG, DAG_prob = sample_DAG(R, scores, max_indegree)
                    update_stats(B_prob, DAG_prob, B, DAG, None, None, "identical_" + move.__name__, None, None)
                    if print_steps:
                        print_step()
                    continue

                t_psum = time.process_time()
                tau_hat_prime = parentsums(B_prime, M, max_indegree, scores)
                t_psum = time.process_time() - t_psum

                R_prob = pi_R(R, scores, max_indegree)
                R_prime_prob = pi_R(R_prime, scores, max_indegree)

                acc_prob = np.exp(R_prime_prob - R_prob)*q_rev/q

            if np.random.rand() < acc_prob:

                t_pos = time.process_time()
                g_prime = pi_B(B_prime, M, max_indegree, scores, tau_hat_prime, return_all=True)
                B_prime_prob = g_prime[0][frozenset()][frozenset()]
                t_pos = time.process_time() - t_pos

                B = B_prime
                tau_hat = tau_hat_prime
                g = g_prime
                R = generate_partition(B, M, tau_hat, g, scores)
                DAG, DAG_prob = sample_DAG(R, scores, max_indegree)
                update_stats(B_prime_prob, DAG_prob, B_prime, DAG, acc_prob, 1, move.__name__, t_psum, t_pos)

            else:
                R = generate_partition(B, M, tau_hat, g, scores)
                DAG, DAG_prob = sample_DAG(R, scores, max_indegree)
                update_stats(B_prob, DAG_prob, B, DAG, acc_prob, 0, move.__name__, t_psum, t_pos)

        if print_steps:
            print_step()

    return stats


def generate_partition(B, M, tau_hat, g, pi_v):

    l = len(B)
    B = [frozenset()] + B + [frozenset()]
    R = list()

    j = 0
    D = frozenset()
    T = frozenset()
    k = 0

    p = dict()
    f = dict()

    while j <= l:
        if D == B[j]:
            j_prime = j+1
            D_prime = frozenset()
        else:
            j_prime = j
            D_prime = D

        if (D.issubset(B[j]) and D != B[j]) or j == l or len(B[j+1]) <= M:
            S = [frozenset(Si) for Si in subsets(B[j_prime].difference(D_prime), 1, max(1, len(B[j_prime].difference(D_prime))))]
            A = frozenset()
        else:
            S = [B[j+1]]
            A = B[j+1].difference({min(B[j+1])})

        for v in B[j_prime].difference(D_prime):
            if not T:
                p[v] = pi_v[v][frozenset()]
            else:
                if T == B[j]:
                    p[v] = tau_hat[v][frozenset({v})]
                else:
                    p[v] = logminus(tau_hat[v][D], tau_hat[v][D.difference(T)])

        f[A] = 0
        for v in A:
            f[A] += p[v]

        # https://stats.stackexchange.com/questions/234544/from-uniform-distribution-to-exponential-distribution-and-vice-versa
        r = -1*(np.random.exponential() - g[j][D][T])
        s = -float('inf')
        for Si in S:
            v = min(Si)
            f[Si] = f[Si.difference({v})] + p[v]
            if D != B[j] or j == 0 or len(Si) > M - len(B[j]):
                s = np.logaddexp(s, f[Si] + g[j_prime][D_prime.union(Si)][Si])
            if s > r:
                k = k + 1
                R.append(Si)
                T = Si
                D = D_prime.union(Si)
                break
        j = j_prime

    return R


def partition(matrix):
    """Partition from adjacency matrix
    """
    partitions = list()
    indices = np.arange(0, len(matrix))
    while True:
        pmask = np.sum(matrix, axis=0) == 0
        if not any(pmask):
            break
        matrix = matrix[:, pmask == False][pmask == False, :]
        partitions.append(indices[pmask])
        indices = indices[pmask == False]
    return partitions


def DAG_str_to_adjmat(DAG_str):

    node_psets = [np.array([int(node) for node in node_pset.split()])
                  for node_pset in DAG_str.split("|")]

    min_node = min([min(node_pset) for node_pset in node_psets])
    max_node = max([max(node_pset) for node_pset in node_psets])

    if min_node == 1:
        node_psets = [node_pset - 1 for node_pset in node_psets]
    else:
        max_node += 1

    matrix = np.zeros((max_node, max_node))

    for node_pset in node_psets:
        node = node_pset[0]
        parents = node_pset[1:]
        for parent in parents:
            matrix[parent, node] = 1

    return matrix


def edge_posteriors(pset_posteriors_path):

    posteriors = read_jkl(pset_posteriors_path)
    for node in posteriors:
        normalizer = np.logaddexp.reduce(list(posteriors[node].values()))
        for pset in posteriors[node]:
            posteriors[node][pset] -= normalizer

    n_vars = len(posteriors.keys())

    if min(posteriors.keys()) == 1:
        adj = -1
    else:
        adj = 0

    edge_posteriors = dict({v: dict({v: list() for v in range(n_vars)}) for v in range(n_vars)})

    for child in posteriors:
        for pset in posteriors[child]:
            for parent in pset:
                edge_posteriors[parent + adj][child + adj].append(posteriors[child][pset])

    for parent in range(n_vars):
        for child in range(n_vars):
            edge_posteriors[parent][child] = np.logaddexp.reduce(edge_posteriors[parent][child])

    edge_posteriors_matrix = np.zeros((n_vars, n_vars))

    for parent in edge_posteriors:
        for child in edge_posteriors[parent]:
            edge_posteriors_matrix[parent, child] = edge_posteriors[parent][child]

    return np.exp(edge_posteriors_matrix)


def edge_empirical_prob_max_error(chainpath, dag_col, exact_probs):
    max_abs_errors = list()
    with open(chainpath) as f:
        n = 0
        DAGs = [DAG_str_to_adjmat(row.strip().split(",")[dag_col]) for row in f.readlines()]
        DAG_sum = np.zeros(DAGs[0].shape)
        for DAG in DAGs:
            n += 1
            DAG_sum += DAG
            max_abs_errors.append(np.max(np.abs(DAG_sum/n - exact_probs)))
    return max_abs_errors


def analyze_chain(chainpath, dag_col):
    with open(chainpath) as f:
        R = [partition(DAG) for DAG in [DAG_str_to_adjmat(row.strip().split(",")[dag_col]) for row in f.readlines()]]
        k = [len(Ri) for Ri in R]
        mu = [len(Rij) for Ri in R for Rij in Ri]
        ell = [len(R_to_B(Ri, 9)) for Ri in R]
        return np.mean(k), np.mean(mu), np.mean(ell)
