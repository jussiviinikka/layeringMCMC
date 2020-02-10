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
            # if len(B[j]) == 1 part disappears
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


def relocate_uniform(B, M):
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

    b = [len(part) for part in B]
    possible_sources = valid_sources(b, M)
    source = np.random.choice(np.nonzero(possible_sources)[0])
    possible_targets = valid_targets(b, source, M)
    target = np.random.choice(np.nonzero(possible_targets)[0])
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


def MCMC(M, iterations, max_indegree, scores, return_all=False, print_steps=False, seed=None):

    if seed is not None:
        np.random.seed(seed)

    B = map_names(list(scores.keys()), np.random.choice(list(gen_M_layerings(len(scores.keys()), M)), 1)[0])

    def print_step(del_first=True):
        items = [
            B_probs,
            acceptance_probs,
            DAG_probs,
            t["parentsums"],
            t["posterior"],
            Bs,
            DAGs,
        ]
        print(",".join(["{}"]*len(items)).format(*([item[-1] for item in items[:-2]] + [structure_to_str(item[-1]) for item in items[-2:]])))
        if del_first:
            for item in items:
                del item[0]

    B_probs = list()
    Bs = list()
    DAG_probs = list()
    DAGs = list()
    acceptance_probs = list()

    # For reporting time used by various functions
    t = dict()
    t["parentsums"] = list()
    t["posterior"] = list()

    stay_prob = 0.01

    t["parentsums"].append(time.process_time())
    tau_hat = parentsums(B, M, max_indegree, scores)
    t["parentsums"][-1] = time.process_time() - t["parentsums"][-1]

    t["posterior"].append(time.process_time())
    g = pi_B(B, M, max_indegree, scores, tau_hat, return_all=True)
    prob_B = g[0][frozenset()][frozenset()]
    t["posterior"][-1] = time.process_time() - t["posterior"][-1]

    B_probs.append(prob_B)
    Bs.append(B)
    acceptance_probs.append(1)

    DAG, DAG_prob = sample_DAG(generate_partition(B, M, tau_hat, g, scores), scores, max_indegree)
    DAGs.append(DAG)
    DAG_probs.append(DAG_prob)

    if print_steps:
        print_step(del_first=False)
        
    for i in range(iterations-1):

        if np.random.rand() < stay_prob:
            Bs.append(B)
            B_probs.append(B_probs[-1])
            t["parentsums"].append(0)
            t["posterior"].append(0)
            acceptance_probs.append(1)  # ?
            DAGs.append(DAGs[-1])
            DAG_probs.append(DAG_probs[-1])

            if print_steps:
                print_step()

        else:

            B_prime, q, q_rev = relocate_uniform(B, M)

            t["parentsums"].append(time.process_time())
            tau_hat = parentsums(B_prime, M, max_indegree, scores)
            t["parentsums"][-1] = time.process_time() - t["parentsums"][-1]

            prob_B = B_probs[-1]

            t["posterior"].append(time.process_time())
            g = pi_B(B_prime, M, max_indegree, scores, tau_hat, return_all=True)
            prob_B_prime = g[0][frozenset()][frozenset()]
            t["posterior"][-1] = time.process_time() - t["posterior"][-1]

            acceptance_probs.append(min(1, np.exp(prob_B_prime - prob_B)*q_rev/q))
            if np.random.rand() < acceptance_probs[-1]:
                Bs.append(B_prime)
                B_probs.append(prob_B_prime)
                B = B_prime

                DAG, DAG_prob = sample_DAG(generate_partition(B, M, tau_hat, g, scores), scores, max_indegree)
                DAGs.append(DAG)
                DAG_probs.append(DAG_prob)

            else:
                Bs.append(B)
                B_probs.append(prob_B)

                DAGs.append(DAGs[-1])
                DAG_probs.append(DAG_probs[-1])

            if print_steps:
                print_step()

    if return_all:
        return B_probs, Bs, acceptance_probs, t, DAG_probs
    return B_probs, Bs


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

